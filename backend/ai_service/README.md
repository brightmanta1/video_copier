# AI-сервис для Video Copier

Модуль искусственного интеллекта для анализа и обработки видео.

## Возможности

- Обнаружение эффектов в видео (переходы, фильтры и т.д.)
- Классификация типов кадров
- Анализ сцен и переходов
- Распознавание шаблонов монтажа

## Требования

Для работы AI-сервиса требуется одно из следующих:

1. **Python с установленным TensorFlow**:
   - Python 3.10 или ниже (TensorFlow не поддерживает Python 3.11+)
   - TensorFlow 2.9+
   - Необходимые библиотеки из `requirements.txt`

2. **Docker** (рекомендуется):
   - Docker Desktop для Windows/macOS
   - Образ с TensorFlow (создается автоматически)

## Установка

### Вариант 1: Локальная установка

```bash
# Установка зависимостей
pip install -r backend/ai_service/requirements.txt

# Обучение моделей
python backend/ai_service/run_training.py
```

### Вариант 2: Использование Docker (рекомендуется)

```bash
# Windows
.\run-tensorflow.bat

# Linux/macOS
./scripts/docker-run.sh
```

## Интеграция с проектом

AI-сервис автоматически определяет доступность TensorFlow:

1. Сначала проверяется локальная установка TensorFlow
2. Если локальная установка отсутствует, используется Docker
3. Если Docker недоступен, функции AI отключаются с соответствующими предупреждениями

### Использование в коде

```python
from backend.ai_service import effect_detector

# Анализ изображения на наличие эффектов
result = effect_detector.predict(image)
print(f"Обнаружен эффект: {result['effect']} с уверенностью {result['confidence']}")
```

## Обученные модели

Обученные модели хранятся в директории `backend/ai_service/trained_models`.
При первом запуске будут созданы простые модели, которые можно улучшить, запустив полное обучение:

```bash
# Локально
python backend/ai_service/run_training.py --dataset Edit3K

# Через Docker
docker-compose run train python run_training.py --dataset Edit3K
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

## Решение проблем

### TensorFlow недоступен

Если вы видите сообщение "TensorFlow недоступен", установите Docker Desktop и запустите:

```bash
.\run-tensorflow.bat  # для Windows
```

### Ошибки с GPU

Для использования GPU с TensorFlow через Docker:

1. Установите драйверы NVIDIA
2. Установите NVIDIA Docker Toolkit
3. Запустите с флагом GPU:

```bash
docker-compose run --gpus all tensorflow bash
``` 