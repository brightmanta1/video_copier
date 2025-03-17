import os
import logging
from typing import Dict, Any

logger = logging.getLogger('Config')

class AppConfig:
    """
    Основная конфигурация приложения
    """
    # Общие настройки
    DEBUG = True
    SECRET_KEY = 'video_copier_secret_key'
    
    # Директории
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'output'
    CACHE_FOLDER = 'cache'
    DATASETS_FOLDER = 'datasets'
    
    # Настройки сервера
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Настройки API
    API_PREFIX = '/api/v1'
    CORS_ORIGINS = ['http://localhost:4200', 'http://127.0.0.1:4200']
    
    # Параметры загрузки файлов
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB 
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv'}
    
    # Параметры кеширования
    ENABLE_CACHE = True
    CACHE_MAX_AGE_DAYS = 7
    CACHE_MAX_SIZE_MB = 5000
    
    # Параметры оптимизации
    ENABLE_MEMORY_OPTIMIZATION = True
    MEMORY_LIMIT_PERCENT = 80.0
    
    # Оптимизация датасетов
    ENABLE_DATASET_OPTIMIZATION = True
    
    # Использование GPU
    USE_GPU = True
    
    def __init__(self):
        # Создаем необходимые директории
        for folder in [self.UPLOAD_FOLDER, self.OUTPUT_FOLDER, self.CACHE_FOLDER, self.DATASETS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
    
    def get_dict(self) -> Dict[str, Any]:
        """Получает словарь с конфигурацией"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k.isupper()}


class VideoAnalyzerConfig:
    """
    Конфигурация анализатора видео
    """
    # Параметры обнаружения границ сцен
    SHOT_THRESHOLD = 30
    
    # Параметры извлечения кадров
    FRAME_SAMPLE_RATE = 1  # каждый кадр
    
    # Параметры характеристик кадра
    FRAME_WIDTH = 640
    
    # Параметры модели
    MODEL_PATH = 'models/video_analyzer_model.h5'
    
    # Оптимизация
    USE_MEMORY_OPTIMIZATION = True
    USE_CACHE = True
    
    # Производительность
    BATCH_SIZE = 32


class DatasetConfig:
    """
    Конфигурация работы с датасетами
    """
    # Пути к датасетам
    AVE_PATH = 'datasets/AVE'
    EDIT3K_PATH = 'datasets/Edit3K'
    
    # Оптимизация
    USE_OPTIMIZATION = True
    USE_CACHE = True


class TrainingConfig:
    """
    Конфигурация для обучения модели
    """
    # Параметры обучения
    EPOCHS = 20
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Параметры модели
    MODEL_TYPE = 'cnn'  # cnn, lstm, transformer
    
    # Оптимизация
    USE_MIXED_PRECISION = True
    USE_GPU = True
    
    # Сохранение
    MODEL_SAVE_PATH = 'models/trained_model.h5'
    CHECKPOINT_DIR = 'models/checkpoints'


class EffectsConfig:
    """
    Конфигурация эффектов
    """
    # Директория с шаблонами эффектов
    EFFECTS_TEMPLATES_DIR = 'effects/templates'
    
    # Параметры качества
    OUTPUT_QUALITY = 'high'  # low, medium, high
    
    # Параметры рендеринга
    USE_GPU_RENDERING = True
    THREADS = 4


class ApiConfig:
    """
    Конфигурация API
    """
    # Лимиты запросов
    RATE_LIMIT = '60 per minute'
    
    # Аутентификация
    AUTH_REQUIRED = False
    JWT_SECRET_KEY = 'jwt_secret_key'
    
    # CORS
    CORS_ORIGINS = ['http://localhost:4200', 'http://127.0.0.1:4200']


def get_config() -> AppConfig:
    """Получает конфигурацию приложения"""
    return AppConfig()


def get_analyzer_config() -> VideoAnalyzerConfig:
    """Получает конфигурацию анализатора видео"""
    return VideoAnalyzerConfig()


def get_dataset_config() -> DatasetConfig:
    """Получает конфигурацию датасетов"""
    return DatasetConfig()


def get_training_config() -> TrainingConfig:
    """Получает конфигурацию обучения"""
    return TrainingConfig()


def get_effects_config() -> EffectsConfig:
    """Получает конфигурацию эффектов"""
    return EffectsConfig()


def get_api_config() -> ApiConfig:
    """Получает конфигурацию API"""
    return ApiConfig() 