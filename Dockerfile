FROM python:3.10-slim

# Инсталирај системски аудио пакети И алатки за компајлирање
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Прво го надградуваме pip
RUN pip install --no-cache-dir --upgrade pip

# Копирај ги барањата и инсталирај ги
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирај го останатиот код
COPY . .

# Стартувај на порта 7860
CMD ["python", "app.py"]