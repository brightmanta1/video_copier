"""
Video Copier - система для анализа и копирования структуры видео.
"""

__version__ = '0.1.0'

# Инициализация базового логирования
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('video_copier')
logger.info(f"Инициализация Video Copier версии {__version__}")

# Настройка обработки исключений импорта
import os
import sys

def create_path_structure():
    """Создает необходимую структуру каталогов"""
    paths = [
        'datasets/AVE',
        'datasets/Edit3K',
        'output',
        'cache',
        'logs'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    logger.info("Структура каталогов создана")

create_path_structure()

# Добавляем текущий каталог в путь поиска модулей
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.debug(f"Добавлен каталог в sys.path: {parent_dir}")

# Проверка критичных зависимостей
def check_dependencies():
    """Проверяет наличие основных зависимостей"""
    import platform
    system = platform.system()
    
    logger.info(f"Определена система: {system}")
    
    dependencies = {
        "moviepy": "Для работы с видеофайлами",
        "ffmpeg-python": "Для высокопроизводительных операций с видео", 
        "opencv-python": "Для анализа видео"
    }
    
    missing = []
    for package, purpose in dependencies.items():
        try:
            if package == "ffmpeg-python":
                # Для ffmpeg-python нужна специальная проверка
                import importlib.util
                if importlib.util.find_spec("ffmpeg") is None:
                    missing.append(f"{package} ({purpose})")
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(f"{package} ({purpose})")
    
    if missing:
        logger.warning("Отсутствуют важные зависимости: " + ", ".join(missing))
        logger.warning("Установите их с помощью: pip install " + " ".join([p.split()[0] for p in missing]))
        return False
    
    # Информируем о TensorFlow
    logger.info("TensorFlow отключен в этой конфигурации. Функции машинного обучения недоступны.")
    
    return True

# Проверяем зависимости
check_dependencies()

# Экспортируем основные компоненты
try:
    from .app import create_app
    from .config import (
        get_config, get_analyzer_config, get_dataset_config,
        get_training_config, get_effects_config, get_api_config
    )
except ImportError as e:
    logger.error(f"Ошибка импорта основных компонентов: {e}")
    
logger.debug("Инициализация пакета завершена") 