FROM python:3.10-slim

# Системски пакети + алатки за аудио и компајлирање
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    gcc \
    python3-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Надградба на pip и инсталација на подготвителни алатки
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Порта 7860 за Hugging Face
EXPOSE 7860

CMD ["python", "app.py"]

# ... претходниот дел од Dockerfile ...

COPY requirements.txt .
# Користиме --no-cache-dir за да заштедиме простор и ги инсталираме torch прво
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
# ... остатокот ...