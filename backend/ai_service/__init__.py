"""
AI Service для Video Copier - модуль для анализа и обработки видео
с помощью моделей машинного обучения.
"""

__version__ = '0.1.0'

import os
import sys
import logging
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_service')

# Определяем пути для моделей
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"

# Создаем директорию для обученных моделей, если она не существует
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Проверяем доступность TensorFlow
try:
    import tensorflow as tf
    logger.info(f"TensorFlow доступен локально, версия: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow не удалось импортировать напрямую, проверяем Docker...")
    
    # Пытаемся импортировать модуль Docker-TensorFlow
    try:
        # Добавляем корневую директорию проекта в путь для импорта
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        # Импортируем модуль Docker-TensorFlow
        from backend.python.app.utils.tensorflow_docker import import_tensorflow, tensorflow_docker
        
        # Проверяем доступность Docker
        if tensorflow_docker.is_tensorflow_available():
            tf = import_tensorflow()
            if tf is not None:
                logger.info("TensorFlow доступен через Docker")
                TENSORFLOW_AVAILABLE = True
            else:
                logger.error("TensorFlow недоступен ни локально, ни через Docker")
                TENSORFLOW_AVAILABLE = False
        else:
            logger.error("Docker с TensorFlow недоступен")
            TENSORFLOW_AVAILABLE = False
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля Docker-TensorFlow: {str(e)}")
        TENSORFLOW_AVAILABLE = False

# Проверяем наличие обученных моделей
def check_trained_models():
    """Проверяет наличие обученных моделей в директории trained_models"""
    model_files = list(MODELS_DIR.glob('*.h5'))
    
    if model_files:
        logger.info(f"Найдено {len(model_files)} обученных моделей: {', '.join([m.name for m in model_files])}")
        return True
    else:
        logger.warning("Обученные модели не найдены")
        return False

# Импортируем модели, только если TensorFlow доступен
if TENSORFLOW_AVAILABLE:
    try:
        from backend.ai_service.models.effect_detector import effect_detector
        from backend.ai_service.models.example_model import example_model
        logger.info("Модели успешно импортированы")
    except Exception as e:
        logger.error(f"Ошибка при импорте моделей: {str(e)}")
else:
    logger.warning("TensorFlow недоступен, модели не импортированы")

# Экспортируем основные классы
try:
    from .models.shot_classifier import ShotClassifier
    from .models.effect_detector import EffectDetector
except ImportError as e:
    logger.warning(f"Не удалось импортировать модели: {e}") 