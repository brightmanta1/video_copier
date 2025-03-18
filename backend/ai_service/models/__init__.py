import os
import sys
import logging
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_models')

# Определяем пути для моделей
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"

# Создаем директорию для обученных моделей, если она не существует
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Добавляем корневую директорию проекта в путь для импорта
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Проверяем наличие TensorFlow
TENSORFLOW_AVAILABLE = False

# Пробуем импортировать через Docker
try:
    from backend.python.app.utils.tensorflow_docker import import_tensorflow
    tf = import_tensorflow()
    if tf is not None:
        logger.info(f"TensorFlow загружен через Docker, версия: {tf.__version__}")
        TENSORFLOW_AVAILABLE = True
    else:
        logger.error("Не удалось загрузить TensorFlow через Docker")
        TENSORFLOW_AVAILABLE = False
except ImportError as e:
    logger.error(f"Ошибка импорта модуля Docker-TensorFlow: {str(e)}")
    
    # Дополнительная информация для диагностики
    if os.system("docker ps") == 0:
        logger.info("Docker запущен, но модуль TensorFlow недоступен")
        running_containers = os.popen("docker ps | grep tensorflow").read().strip()
        if running_containers:
            logger.info(f"Найдены запущенные контейнеры TensorFlow:\n{running_containers}")
        else:
            logger.info("Контейнеры TensorFlow не найдены")
    else:
        logger.error("Docker не доступен или не запущен")
    
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

# Функция для загрузки моделей
def load_model(model_path, custom_objects=None):
    """
    Загружает модель через Docker TensorFlow
    
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
        logger.error("TensorFlow через Docker недоступен, загрузка модели невозможна")
        return None
        
    try:
        # Загружаем модель через Docker
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Модель успешно загружена через Docker: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели через Docker {model_path}: {str(e)}")
        return None

# Импортируем модели, только если TensorFlow через Docker доступен
if TENSORFLOW_AVAILABLE:
    try:
        # Импортируем только если файлы существуют
        effect_detector_path = BASE_DIR / "effect_detector.py"
        shot_classifier_path = BASE_DIR / "shot_classifier.py"
        
        if effect_detector_path.exists():
            try:
                from .effect_detector import EffectDetector
                logger.info("Модель EffectDetector успешно импортирована")
            except Exception as e:
                logger.error(f"Ошибка при импорте EffectDetector: {str(e)}")
        
        if shot_classifier_path.exists():
            try:
                from .shot_classifier import ShotClassifier
                logger.info("Модель ShotClassifier успешно импортирована")
            except Exception as e:
                logger.error(f"Ошибка при импорте ShotClassifier: {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка при импорте моделей: {str(e)}")
else:
    logger.warning("TensorFlow через Docker недоступен, модели не импортированы")
    
    # Создаем заглушки для моделей
    class DummyModel:
        """Заглушка для моделей, когда TensorFlow через Docker недоступен"""
        def __init__(self, *args, **kwargs):
            logger.warning("Создана заглушка модели (TensorFlow через Docker недоступен)")
            
        def predict(self, *args, **kwargs):
            logger.error("Невозможно выполнить предсказание: TensorFlow через Docker недоступен")
            return None
            
        def train(self, *args, **kwargs):
            logger.error("Невозможно обучить модель: TensorFlow через Docker недоступен")
            return None
    
    # Создаем заглушки для классов моделей
    class EffectDetector(DummyModel):
        pass
        
    class ShotClassifier(DummyModel):
        pass 