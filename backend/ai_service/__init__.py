"""
AI Service для Video Copier - модуль для анализа и обработки видео
с помощью моделей машинного обучения.
"""

__version__ = '0.1.0'

import os
import sys
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_service')

# Определяем пути к моделям
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')

# Создаем директорию для обученных моделей, если ее нет
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# Проверяем наличие обученных моделей
SHOT_CLASSIFIER_PATH = os.path.join(TRAINED_MODELS_DIR, 'shot_classifier_model.h5')
EFFECT_DETECTOR_PATH = os.path.join(TRAINED_MODELS_DIR, 'effect_detector_model.h5')

# Проверяем доступность моделей и выводим соответствующие сообщения
if os.path.exists(SHOT_CLASSIFIER_PATH):
    logger.info(f"Модель ShotClassifier доступна: {SHOT_CLASSIFIER_PATH}")
else:
    logger.warning(f"Модель ShotClassifier не найдена. Выполните обучение: python backend/ai_service/run_training.py --train-shot-only")

if os.path.exists(EFFECT_DETECTOR_PATH):
    logger.info(f"Модель EffectDetector доступна: {EFFECT_DETECTOR_PATH}")
else:
    logger.warning(f"Модель EffectDetector не найдена. Выполните обучение: python backend/ai_service/run_training.py --train-effect-only")

# Проверяем доступность TensorFlow
try:
    import tensorflow as tf
    logger.info(f"TensorFlow доступен, версия: {tf.__version__}")
except ImportError:
    logger.warning("TensorFlow не установлен. Модели машинного обучения недоступны.")
    logger.warning("Для использования моделей установите TensorFlow: pip install tensorflow")

# Экспортируем основные классы
try:
    from .models.shot_classifier import ShotClassifier
    from .models.effect_detector import EffectDetector
except ImportError as e:
    logger.warning(f"Не удалось импортировать модели: {e}") 