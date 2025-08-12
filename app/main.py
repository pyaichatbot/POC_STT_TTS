
import io
import os
import math
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Response, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Try to import KittenTTS; fall back to a tone generator if not available
try:
    import kittentts as ktts  # type: ignore
    HAS_KITTEN = True
except Exception:
    ktts = None
    HAS_KITTEN = False

# STT with Faster-Whisper (CPU)
from faster_whisper import WhisperModel

app = FastAPI(title="Local TTS/STT PoC", version="0.2.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'index.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.get('/healthz')
async def healthz():
    return {"status": "ok", "tts": HAS_KITTEN}

class TTSIn(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0

# Lazy singletons so container starts fast
_tts_engine = None
_stt_model = None

def get_tts_engine():
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    if HAS_KITTEN:
        # Always load KittenTTS from the local directory for offline use
        local_kitten_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/kittentts'))
        if not os.path.exists(local_kitten_path):
            raise RuntimeError(f"KittenTTS model directory not found at {local_kitten_path}. Please ensure the model files are present for offline use.")
        _tts_engine = ktts.TTS(local_kitten_path)
        return _tts_engine
    else:
        _tts_engine = 'tone'  # placeholder
        return _tts_engine

def synthesize_tone(text: str, speed: float = 1.0) -> bytes:
    # Generates a short sine wave as placeholder (not real TTS)
    sr = 16000
    duration = min(max(len(text) / 20.0, 0.6), 3.0) / max(speed, 0.25)
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = 440.0
    audio = 0.2 * np.sin(2 * math.pi * freq * t)
    audio_i16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    return buf.getvalue()

@app.post('/api/tts')
async def tts(inp: TTSIn):
    eng = get_tts_engine()
    if HAS_KITTEN:
        try:
            wav_bytes = eng.tts_bytes(inp.text, voice=inp.voice, speed=inp.speed)
        except AttributeError:
            try:
                audio, sr = eng.tts(inp.text, voice=inp.voice, speed=inp.speed)
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1 if audio.ndim == 1 else audio.shape[0])
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    if audio.dtype != np.int16:
                        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    wf.writeframes(audio.tobytes())
                wav_bytes = buf.getvalue()
            except Exception as e:
                raise HTTPException(500, f"KittenTTS synthesis failed: {e}")
        return Response(content=wav_bytes, media_type='audio/wav')
    else:
        data = synthesize_tone(inp.text, speed=inp.speed or 1.0)
        return Response(content=data, media_type='audio/wav')

# ---- STT ----
def get_stt_model():
    global _stt_model
    if _stt_model is not None:
        return _stt_model
    model_dir = os.environ.get('WHISPER_MODEL_DIR', './models/whisper-small')
    if not os.path.exists(model_dir):
        raise HTTPException(500, f"Whisper model not found at {model_dir}. Set WHISPER_MODEL_DIR.")
    _stt_model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    return _stt_model

@app.post('/api/stt')
async def stt(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, 'No file provided')
    audio_bytes = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        model = get_stt_model()
        segments, info = model.transcribe(tmp_path, language='en')
        text = ''.join(s.text for s in segments)
        return {"text": text, "duration": getattr(info, 'duration', None)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# --- OpenAI-compatible endpoints ---
class OpenAISpeechIn(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"   # we return WAV
    speed: Optional[float] = 1.0

@app.post("/v1/audio/speech")
async def openai_audio_speech(inp: OpenAISpeechIn):
    if inp.response_format and inp.response_format.lower() != "wav":
        raise HTTPException(400, "Only 'wav' response_format is supported in this PoC.")
    return await tts(TTSIn(text=inp.input, voice=inp.voice, speed=inp.speed))

@app.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(file: UploadFile = File(...), model: str = Form("whisper-small")):
    res = await stt(file)
    return {"text": res.get("text", ""), "model": model}
