FROM python:3.10-slim

# Инсталирај системски аудио пакети
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Прво инсталирај ги библиотеките (за кеширање)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирај го кодот
COPY . .

# Стартувај ја апликацијата на порта 7860
CMD ["python", "app.py"]