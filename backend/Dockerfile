FROM python:3.11-slim

# System libraries required by:
#   - soundfile (libsndfile1) — without this, `import soundfile` fails entirely
#   - opencv-python-headless (libglib2.0-0) — some cv2 functions need it even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deliberately using requirements-deploy.txt (excludes easyocr/torch) to stay
# under Render's 512MB RAM ceiling. agents.py's check_currency() already
# falls back cleanly to the trained visual classifier (91.3% held-out
# accuracy) when the OCR stage is unavailable — no code changes needed.
# Run the full 3-stage version (with OCR) locally for your demo video.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Render sets $PORT at runtime — do not hardcode 8000 here.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
