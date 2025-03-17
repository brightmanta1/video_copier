import os
import uuid
import logging
import importlib.util

logger = logging.getLogger('VideoAnalyzerService')

# Импортируем классы с проверкой и обработкой исключений
try:
    from ..models.video_analyzer import VideoAnalyzer
    from ..models.dataset_manager import DatasetManager
    from ..models.video_editor import VideoEditor
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    # Заглушки на случай, если импорты не удались
    VideoAnalyzer = None
    DatasetManager = None
    VideoEditor = None

class VideoAnalyzerService:
    """Сервис для анализа видео и применения редакторских структур"""
    
    def __init__(self):
        # Проверка успешности импорта необходимых классов
        if None in (VideoAnalyzer, DatasetManager, VideoEditor):
            logger.error("Невозможно инициализировать сервис - отсутствуют необходимые зависимости")
            return
            
        # Инициализация датасет-менеджера
        try:
            self.dataset_manager = DatasetManager(
                ave_path=os.environ.get('AVE_PATH', 'datasets/AVE'),
                edit3k_path=os.environ.get('EDIT3K_PATH', 'datasets/Edit3K')
            )
            
            self.dataset_manager.load_datasets()
        except Exception as e:
            logger.warning(f"Не удалось загрузить датасеты: {e}")
            self.dataset_manager = None
            
        # Инициализация анализатора и редактора
        try:
            self.video_analyzer = VideoAnalyzer(self.dataset_manager)
            self.video_editor = VideoEditor(output_dir='output')
            logger.info("VideoAnalyzerService успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации компонентов: {e}")
            self.video_analyzer = None
            self.video_editor = None
        
    def analyze_video(self, video_path):
        """Анализирует видео и определяет его структуру редактирования"""
        if not self.video_analyzer:
            raise ValueError("Анализатор видео не инициализирован")
            
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
        logger.info(f"Анализ видео: {video_path}")
        return self.video_analyzer.analyze_video(video_path)
    
    def apply_structure(self, target_video_path, edit_structure):
        """Применяет структуру редактирования к целевому видео"""
        if not self.video_editor:
            raise ValueError("Редактор видео не инициализирован")
            
        if not os.path.exists(target_video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {target_video_path}")
            
        logger.info(f"Применение структуры редактирования к: {target_video_path}")
        return self.video_editor.apply_structure(target_video_path, edit_structure)
    
    def get_output_dir(self):
        """Возвращает директорию для результатов редактирования"""
        if not self.video_editor:
            return os.environ.get('OUTPUT_DIR', 'output')
        return self.video_editor.output_dir 