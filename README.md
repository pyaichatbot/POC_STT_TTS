# Local TTS/STT PoC (FastAPI)

**What this is**
- A self-contained FastAPI service exposing:
  - `/api/tts` – Text → Speech (KittenTTS if available, else a WAV tone placeholder)
  - `/api/stt` – Speech → Text using Faster-Whisper (CPU) and a local model path
  - `/v1/audio/speech` – OpenAI-compatible TTS (returns WAV)
  - `/v1/audio/transcriptions` – OpenAI-like STT
  - `/healthz` and a minimal web UI `/` for demos

**How to run locally**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install kittentts wheel if you have it locally
# pip install ./kittentts-0.1.0-py3-none-any.whl

export WHISPER_MODEL_DIR=./models/whisper-small
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Offline/Proxy notes**
- Build the Docker image with a local KittenTTS wheel and your Whisper model copied into the image; then no egress is needed.

**API**
- POST `/api/tts` JSON: `{"text":"Hello","voice":"female_1","speed":1.0}` → `audio/wav`
- POST `/api/stt` multipart: `file=@sample.wav` → `{"text":"..."}`
- POST `/v1/audio/speech` JSON: `{"model":"kittentts-0.1","voice":"female_1","input":"Hello","response_format":"wav"}` → `audio/wav`
- POST `/v1/audio/transcriptions` multipart: `file=@sample.wav` → `{"text":"..."}`

### Sample calls
```bash
curl http://localhost:8080/v1/audio/speech   -H "Content-Type: application/json"   -d '{"model":"kittentts-0.1","voice":"female_1","input":"Hello from KittenTTS PoC","response_format":"wav"}'   --output speech.wav

curl -X POST http://localhost:8080/v1/audio/transcriptions   -F "file=@speech.wav"
```
