FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. CPU верзии на torch
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Инсталирај ги останатите
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. КРИТИЧНО: Верзија 0.24.0 е компатибилна со сите страни
RUN pip install --no-cache-dir --force-reinstall huggingface_hub==0.24.0 gradio_client==1.3.0

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]