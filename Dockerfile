FROM tensorflow/tensorflow:2.12.0-gpu

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Создание директории проекта
WORKDIR /app

# Создадим директории если они не существуют
RUN mkdir -p backend/ai_service
RUN mkdir -p backend/python

# Копирование requirements
COPY backend/python/requirements.txt backend/python/requirements.txt
COPY backend/ai_service/requirements.txt backend/ai_service/requirements.txt

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r backend/python/requirements.txt
RUN pip install --no-cache-dir -r backend/ai_service/requirements.txt

# Копирование исходного кода проекта
COPY . .

# Создание директории для моделей
RUN mkdir -p backend/ai_service/trained_models

# Установка переменных окружения
ENV PYTHONPATH=/app

# Указание рабочей директории
WORKDIR /app 