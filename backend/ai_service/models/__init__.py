import os
import sys
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в путь для импорта
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_models')

# Пытаемся импортировать tensorflow напрямую
try:
    import tensorflow as tf
    logger.info(f"TensorFlow загружен напрямую, версия: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow не удалось загрузить напрямую, пробуем через Docker...")
    
    try:
        # Импортируем модуль Docker-обертки
        from backend.python.app.utils.tensorflow_docker import import_tensorflow
        tf = import_tensorflow()
        if tf is not None:
            logger.info("TensorFlow загружен через Docker")
            TENSORFLOW_AVAILABLE = True
        else:
            logger.error("Не удалось загрузить TensorFlow через Docker")
            TENSORFLOW_AVAILABLE = False
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля Docker-обертки: {str(e)}")
        TENSORFLOW_AVAILABLE = False

# Определяем директорию для обученных моделей
MODELS_DIR = Path(__file__).parent.parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Функция для проверки и загрузки моделей
def load_model(model_path, custom_objects=None):
    """
    Загружает модель TensorFlow/Keras, с поддержкой загрузки через Docker
    
    Args:
        model_path: путь к файлу модели (.h5)
        custom_objects: пользовательские объекты для загрузки модели
        
    Returns:
        загруженная модель или None, если загрузка не удалась
    """
    if not os.path.exists(model_path):
        logger.error(f"Файл модели не существует: {model_path}")
        return None
        
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow недоступен, загрузка модели невозможна")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Модель успешно загружена: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_path}: {str(e)}")
        return None 