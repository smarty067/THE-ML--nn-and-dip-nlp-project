FROM python:3.11-slim

# Install system requirements including Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements directly
RUN pip install --no-cache-dir \
    gradio \
    openai \
    python-dotenv \
    pytesseract \
    pillow \
    pypdf \
    pdf2image \
    faster-whisper \
    av \
    supabase \
    rapidfuzz \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY . /app

EXPOSE 10000

CMD ["python", "app.py"]
