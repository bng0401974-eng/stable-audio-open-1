FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. Прво инсталирај ги тешките работи (CPU верзии за да не пука меморијата)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Инсталирај ги останатите од requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. ФИНАЛЕН УДАР: Прегази го huggingface_hub со верзијата што работи со Gradio 4
RUN pip install --no-cache-dir --force-reinstall huggingface_hub==0.23.2 gradio_client==0.17.0

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]