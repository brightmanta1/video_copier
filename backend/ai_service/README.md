# AI Service для Video Copier

Данный модуль отвечает за обучение и использование моделей машинного обучения для анализа видео.

## Модели

В проекте используются две основные модели:

1. **ShotClassifier** - классификатор типов кадров (wide shot, medium shot, close-up и т.д.) на основе датасета AVE
2. **EffectDetector** - детектор эффектов и переходов на основе датасета Edit3K

## Датасеты

Для обучения моделей используются следующие открытые датасеты:

- **AVE (Audio-Visual Event)** - [github.com/dawitmureja/AVE](https://github.com/dawitmureja/AVE)
- **Edit3K** - [github.com/GX77/Edit3K](https://github.com/GX77/Edit3K)

## Требования и зависимости

Для работы AI Service требуются следующие библиотеки:

```
tensorflow>=2.9.0 (для Python < 3.12)
opencv-python>=4.7.0.72
scikit-learn>=1.2.2
scipy>=1.10.1
gitpython>=3.1.32
```

Полный список зависимостей доступен в файле `requirements.txt`.

## Установка зависимостей

```bash
# Установка базовых зависимостей
pip install -r backend/ai_service/requirements.txt

# Для Windows с GPU (Python 3.11 и ниже)
pip install tensorflow-directml-plugin
```

## Запуск обучения моделей

1. **Полное обучение** - скачивает датасеты и обучает обе модели:

```bash
python backend/ai_service/run_training.py
```

2. **Обучение только модели типов кадров**:

```bash
python backend/ai_service/run_training.py --train-shot-only
```

3. **Обучение только модели эффектов**:

```bash
python backend/ai_service/run_training.py --train-effect-only
```

4. **Пропуск шага клонирования репозиториев**:

```bash
python backend/ai_service/run_training.py --skip-clone
```

5. **Принудительное повторное клонирование репозиториев**:

```bash
python backend/ai_service/run_training.py --force-clone
```

## Структура директорий

```
backend/ai_service/
├── models/                 # Модели машинного обучения
│   ├── effect_detector.py  # Детектор эффектов в видео
│   ├── shot_classifier.py  # Классификатор типов кадров
│   └── model_trainer.py    # Модуль для обучения моделей
├── trained_models/         # Обученные модели сохраняются здесь
├── run_training.py         # Скрипт для запуска обучения
└── requirements.txt        # Зависимости AI сервиса
```

## Возможные проблемы и их решения

### TensorFlow не устанавливается или не запускается

- **Для Python 3.12 и выше**: TensorFlow официально поддерживается только для Python 3.11 и ниже. Используйте виртуальное окружение с Python 3.11.
- **Для Windows**: На Windows с Python 3.11 и ниже установите tensorflow-cpu и tensorflow-directml-plugin для поддержки GPU.

### Ошибка при загрузке датасетов

- Убедитесь, что у вас настроен Git и есть доступ к интернету
- Проверьте, что репозитории доступны по URL
- При необходимости загрузите датасеты вручную и поместите их в директории `datasets/AVE` и `datasets/Edit3K`

### Недостаточно памяти для обучения

- Уменьшите batch_size при вызове метода train
- Используйте модели с меньшим количеством параметров, изменив архитектуру в файлах моделей 