import logging

logger = logging.getLogger('app.utils')

# Инициализируем модуль для работы с TensorFlow через Docker
try:
    from .tensorflow_docker import import_tensorflow, tensorflow_docker
    logger.info("Модуль Docker TensorFlow доступен")
except ImportError as e:
    logger.warning(f"Не удалось импортировать модуль Docker TensorFlow: {str(e)}")

# Импортируем другие утилиты
from .video_utils import VideoUtils 