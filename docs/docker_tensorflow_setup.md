# Настройка TensorFlow через Docker для Video Copier

Это руководство описывает настройку среды разработки для проекта Video Copier с использованием Docker вместо WSL.

## Шаг 1: Установка Docker Desktop

1. Скачайте Docker Desktop для Windows с [официального сайта](https://www.docker.com/products/docker-desktop/)
2. Установите Docker Desktop, следуя инструкциям установщика
3. После установки перезагрузите компьютер
4. Запустите Docker Desktop и дождитесь полной инициализации

## Шаг 2: Создание Dockerfile для проекта

1. Создайте в корне проекта файл `Dockerfile`:

```dockerfile
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

# Копирование requirements
COPY backend/ai_service/requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта
COPY . .

# Создание директории для моделей
RUN mkdir -p backend/ai_service/trained_models

# Указание рабочей директории
WORKDIR /app/backend/ai_service
```

2. Создайте файл `docker-compose.yml` в корне проекта:

```yaml
version: '3'

services:
  tensorflow:
    build: .
    volumes:
      - .:/app
    ports:
      - "5000:5000"  # Для API
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # Для использования GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

## Шаг 3: Настройка NVIDIA Docker (для GPU)

Если у вас есть GPU NVIDIA и вы хотите использовать его:

1. Убедитесь, что у вас установлены последние драйверы NVIDIA
2. Установите [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):
   - Скачайте и запустите установщик
   - Перезагрузите Docker: `systemctl restart docker` (в терминале администратора)
3. Проверьте работу NVIDIA Docker:
   ```
   docker run --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
   ```

## Шаг 4: Запуск контейнера и обучение моделей

1. Соберите Docker-образ:
   ```
   docker-compose build
   ```

2. Запустите обучение моделей в контейнере:
   ```
   docker-compose run tensorflow python run_training.py
   ```

3. Для запуска интерактивной оболочки:
   ```
   docker-compose run tensorflow bash
   ```

## Шаг 5: Интеграция с Windows-окружением

1. Обученные модели будут доступны в директории проекта благодаря монтированию тома:
   ```
   backend/ai_service/trained_models/
   ```

2. Для запуска Flask API в контейнере:
   ```
   docker-compose run -p 5000:5000 tensorflow python /app/backend/python/run.py
   ```

## Преимущества подхода с Docker

1. **Изоляция**: Полная изоляция среды TensorFlow от вашей основной системы
2. **Воспроизводимость**: Гарантированно одинаковое окружение на всех компьютерах
3. **Простое управление версиями**: Легко переключаться между версиями TensorFlow
4. **Поддержка GPU**: Полноценная поддержка GPU NVIDIA через NVIDIA Container Toolkit
5. **Отсутствие конфликтов**: Нет конфликтов между Python-пакетами в системе

## Решение проблем

### Нет доступа к GPU

Проверьте:
1. Установлен ли NVIDIA Container Toolkit 
2. Содержит ли docker-compose.yml конфигурацию для GPU
3. Выведите статус GPU: `docker-compose run tensorflow nvidia-smi`

### Ошибки при сборке образа

1. Очистите кеш Docker: `docker system prune -a`
2. Проверьте наличие всех зависимостей в requirements.txt
3. Проверьте доступ к интернету для загрузки пакетов

### Медленная работа с файлами

В Windows файловая система Docker может работать медленно:
1. Используйте команду `docker-compose run` с флагом `-v` для создания именованных томов
2. Храните большие наборы данных в именованном томе Docker вместо монтирования локальной директории 