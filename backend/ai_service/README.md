# AI-сервис для Video Copier

Модуль искусственного интеллекта для анализа и обработки видео с использованием TensorFlow через Docker.

## Возможности

- Обнаружение эффектов в видео (переходы, фильтры и т.д.)
- Классификация типов кадров
- Анализ сцен и переходов
- Распознавание шаблонов монтажа

## Требования

Для работы AI-сервиса требуется:

- **Docker** (обязательно)
  - Docker Desktop для Windows/macOS
  - Образ TensorFlow (загружается автоматически)
  - Docker Engine для Linux

> **Важно:** AI-сервис работает **исключительно с Docker**. Локальная установка TensorFlow не используется и не требуется.

## Установка и настройка

```bash
# Windows
.\run-tensorflow.bat

# Linux/macOS
./scripts/docker-run.sh
```

Эти скрипты автоматически:
1. Проверят наличие Docker
2. Загрузят образ TensorFlow, если он не установлен
3. Запустят контейнер с TensorFlow в фоновом режиме
4. Настроят доступ к TensorFlow через Docker API

## Интеграция с проектом

AI-сервис автоматически подключается к Docker при первом импорте:

```python
from backend.ai_service import effect_detector

# Анализ изображения на наличие эффектов
result = effect_detector.predict(image)
print(f"Обнаружен эффект: {result['effect']} с уверенностью {result['confidence']}")
```

Весь процесс работы с TensorFlow через Docker полностью автоматизирован и прозрачен для разработчика.

## Обученные модели

Обученные модели хранятся в директории `backend/ai_service/trained_models`.
При первом запуске будут созданы простые модели, которые можно улучшить, запустив полное обучение:

```bash
# Запуск обучения в Docker
docker exec TensorFlow_CONTAINER_ID python /app/backend/ai_service/run_training.py --dataset Edit3K
```

## Структура директорий

```
backend/ai_service/
├── models/               # Определения моделей
│   ├── __init__.py
│   ├── effect_detector.py
│   └── example_model.py
├── trained_models/       # Обученные модели
│   ├── effect_detector.h5
│   └── example_model.h5
├── __init__.py           # Модуль AI-сервиса
├── requirements.txt      # Зависимости
└── README.md             # Эта документация
```

## Технические детали работы с Docker

AI-сервис использует следующий подход для работы с TensorFlow через Docker:

1. При первом импорте модуля проверяется наличие запущенного контейнера Docker с TensorFlow
2. Если контейнер не запущен, он автоматически запускается
3. Выполнение функций TensorFlow происходит путем передачи кода в контейнер и получения результатов
4. Прокси-классы создают иллюзию работы с обычным TensorFlow API

## Решение проблем

### TensorFlow недоступен

Если вы видите сообщение "TensorFlow через Docker недоступен":

1. Убедитесь, что Docker Desktop запущен
2. Запустите `run-tensorflow.bat` (Windows) или `./scripts/docker-run.sh` (Linux/macOS)
3. Проверьте статус контейнеров: `docker ps | grep tensorflow`

### Ошибки с GPU

Для использования GPU с TensorFlow через Docker:

1. Установите драйверы NVIDIA
2. Установите NVIDIA Docker Toolkit
3. Запустите с флагом GPU:

```bash
docker run --gpus all -d --rm -v "%cd%:/app" tensorflow/tensorflow:2.12.0-gpu ...
``` 