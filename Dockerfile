# Dockerfile for the POC STT TTS application
# This Dockerfile sets up a Python environment with all necessary dependencies
# and copies the application code and model weights for offline use.
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 \
	HF_HOME=/srv/.cache/huggingface \
	WHISPER_MODEL_DIR=/srv/models/whisper-small \
	HF_HUB_DISABLE_TELEMETRY=1

# Optional proxies (uncomment and pass as build-args)
# ARG HTTP_PROXY
# ARG HTTPS_PROXY
# ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY}

COPY requirements.txt ./

# Copy all wheels for offline install
COPY models/whls/*.whl /tmp/whls/

# Install all wheels (dependencies and KittenTTS)
RUN pip install --no-cache-dir /tmp/whls/*.whl

# Copy app code
COPY app ./app

# Copy whisper-small model weights into the image (for full offline)
COPY models/whisper-small /srv/models/whisper-small

# Copy kittentts model weights into the image (for full offline)
COPY models/kittentts /srv/models/kittentts

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
