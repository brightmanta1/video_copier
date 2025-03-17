import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import logging

logger = logging.getLogger('DatasetManager')

# Импортируем утилиты из нашего приложения с проверкой
try:
    from ..utils.dataset_optimizer import DatasetOptimizer
    from ..utils.memory_manager import MemoryManager
    from ..utils.cache_manager import CacheManager
    utils_import_error = None
except ImportError as e:
    utils_import_error = str(e)
    logger.error(f"Ошибка импорта утилит: {e}")
    # Создаем заглушки для отсутствующих классов
    DatasetOptimizer = None
    MemoryManager = None
    CacheManager = None

class DatasetManager:
    """Единый менеджер для работы с AVE и Edit3K датасетами"""
    
    def __init__(self, ave_path: str = "datasets/AVE", edit3k_path: str = "datasets/Edit3K", 
                 use_optimization: bool = True, use_cache: bool = True):
        # Проверяем наличие директорий для датасетов
        for path in [ave_path, edit3k_path]:
            os.makedirs(path, exist_ok=True)
                
        self.ave_path = ave_path
        self.edit3k_path = edit3k_path
        self.ave_data = {}
        self.edit3k_data = {}
        self.effect_types = {}
        self.use_optimization = use_optimization and DatasetOptimizer is not None
        
        # Проверяем наличие ошибок импорта
        if utils_import_error:
            logger.warning(f"Некоторые функции DatasetManager будут недоступны из-за ошибок импорта: {utils_import_error}")
            self.use_optimization = False
            self.optimizer = None
            self.memory_manager = None
            self.cache_manager = None
            return
        
        # Инициализация оптимизатора и менеджера памяти
        if self.use_optimization:
            try:
                self.optimizer = DatasetOptimizer(cache_dir='cache')
                self.memory_manager = MemoryManager()
                # Запускаем мониторинг памяти
                self.memory_manager.start_memory_monitoring()
                logger.info("Оптимизация датасетов включена")
            except Exception as e:
                logger.error(f"Не удалось инициализировать оптимизаторы: {e}")
                self.use_optimization = False
                self.optimizer = None
                self.memory_manager = None
            
        # Инициализация менеджера кеша
        if use_cache and CacheManager is not None:
            try:
                self.cache_manager = CacheManager(cache_dir='cache/datasets')
                logger.info("Кеширование датасетов включено")
            except Exception as e:
                logger.error(f"Не удалось инициализировать менеджер кеша: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None
        
        # Категории эффектов из Edit3K
        self.effect_categories = {
            'video_effects': [],
            'animations': [],
            'transitions': [],
            'filters': [],
            'stickers': [],
            'text': []
        }
        
        # Характеристики съемки из AVE
        self.shot_features = {
            'sizes': set(),
            'angles': set(),
            'types': set(),
            'motions': set()
        }
        
    def load_datasets(self) -> None:
        """Загрузка обоих датасетов"""
        if self.use_optimization:
            self._load_optimized_ave()
            self._load_optimized_edit3k()
        else:
            self._load_ave()
            self._load_edit3k()
            
        print(f"DatasetManager: Загружено {len(self.ave_data)} сцен из AVE и {len(self.edit3k_data)} эффектов из Edit3K")
    
    def _load_optimized_ave(self) -> None:
        """Загрузка оптимизированного AVE датасета"""
        try:
            # Проверяем наличие оптимизированного датасета
            optimized_path = os.path.join('cache', 'ave_optimized.pkl')
            
            if not os.path.exists(optimized_path):
                # Создаем оптимизированный датасет
                optimized_path = self.optimizer.optimize_ave_dataset(self.ave_path)
            
            # Загружаем оптимизированный датасет
            self.ave_data = self.optimizer.load_optimized_ave(optimized_path)
            
            # Сбор уникальных характеристик
            for scene in self.ave_data.values():
                for shot in scene.values():
                    if isinstance(shot.get('shot-size'), list) and len(shot.get('shot-size', [])) > 0:
                        self.shot_features['sizes'].add(shot['shot-size'][0])
                    if isinstance(shot.get('shot-angle'), list) and len(shot.get('shot-angle', [])) > 0:
                        self.shot_features['angles'].add(shot['shot-angle'][0])
                    if isinstance(shot.get('shot-type'), list) and len(shot.get('shot-type', [])) > 0:
                        self.shot_features['types'].add(shot['shot-type'][0])
                    if isinstance(shot.get('shot-motion'), list) and len(shot.get('shot-motion', [])) > 0:
                        self.shot_features['motions'].add(shot['shot-motion'][0])
                        
        except Exception as e:
            print(f"Error loading optimized AVE dataset: {e}")
            # Если не удалось загрузить оптимизированный датасет, пробуем обычный
            self._load_ave()
    
    def _load_optimized_edit3k(self) -> None:
        """Загрузка оптимизированного Edit3K датасета"""
        try:
            # Проверяем наличие оптимизированного датасета
            optimized_path = os.path.join('cache', 'edit3k_optimized.h5')
            
            if not os.path.exists(optimized_path):
                # Создаем оптимизированный датасет
                optimized_path = self.optimizer.optimize_edit3k_dataset(self.edit3k_path)
            
            # Загружаем оптимизированный датасет
            self.edit3k_data, self.effect_types = self.optimizer.load_optimized_edit3k(optimized_path)
            
            # Категоризация эффектов
            for effect_id, effect_data in self.edit3k_data.items():
                effect_type = self.effect_types.get(effect_id)
                if effect_type in self.effect_categories:
                    self.effect_categories[effect_type].append(effect_data)
                    
        except Exception as e:
            print(f"Error loading optimized Edit3K dataset: {e}")
            # Если не удалось загрузить оптимизированный датасет, пробуем обычный
            self._load_edit3k()
        
    def _load_ave(self) -> None:
        """Загрузка AVE dataset"""
        # Проверяем кеш, если доступен
        if self.cache_manager:
            cache_key = f"ave_dataset_{self.ave_path}"
            cached_result = self.cache_manager.get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Используем кешированные данные AVE")
                self.ave_data = cached_result['data']
                # Обновляем характеристики съемки
                self._update_shot_features()
                return
        
        try:
            with open(os.path.join(self.ave_path, 'annotations.json'), 'r') as f:
                self.ave_data = json.load(f)
                
            # Кешируем результат, если доступно
            if self.cache_manager:
                self.cache_manager.cache_result(f"ave_dataset_{self.ave_path}", self.ave_data)
                
            # Обновляем характеристики съемки
            self._update_shot_features()
                    
        except Exception as e:
            logger.error(f"Error loading AVE dataset: {e}")
            
    def _update_shot_features(self) -> None:
        """Обновляет характеристики съемки на основе данных AVE"""
        # Сбор уникальных характеристик
        for scene in self.ave_data.values():
            for shot in scene.values():
                if isinstance(shot.get('shot-size'), list) and len(shot.get('shot-size', [])) > 0:
                    self.shot_features['sizes'].add(shot['shot-size'][0])
                if isinstance(shot.get('shot-angle'), list) and len(shot.get('shot-angle', [])) > 0:
                    self.shot_features['angles'].add(shot['shot-angle'][0])
                if isinstance(shot.get('shot-type'), list) and len(shot.get('shot-type', [])) > 0:
                    self.shot_features['types'].add(shot['shot-type'][0])
                if isinstance(shot.get('shot-motion'), list) and len(shot.get('shot-motion', [])) > 0:
                    self.shot_features['motions'].add(shot['shot-motion'][0])
    
    def _load_edit3k(self) -> None:
        """Загрузка Edit3K dataset"""
        try:
            edit3k_path = os.path.join(self.edit3k_path, 'edit_3k.json')
            id2type_path = os.path.join(self.edit3k_path, 'edit_id2type.json')
            
            if os.path.exists(edit3k_path) and os.path.exists(id2type_path):
                with open(edit3k_path, 'r') as f:
                    self.edit3k_data = json.load(f)
                
                # Загрузка маппинга эффектов
                with open(id2type_path, 'r') as f:
                    self.effect_types = json.load(f)
                    
                # Категоризация эффектов
                for effect_id, effect_data in self.edit3k_data.items():
                    effect_type = self.effect_types.get(effect_id)
                    if effect_type in self.effect_categories:
                        self.effect_categories[effect_type].append(effect_data)
            else:
                print(f"Edit3K files not found at {edit3k_path}")
                    
        except Exception as e:
            print(f"Error loading Edit3K dataset: {e}")
            
    def get_shot_features(self, scene_id: str, shot_id: str) -> Optional[Dict]:
        """Получение характеристик конкретного шота"""
        return self.ave_data.get(scene_id, {}).get(shot_id)
    
    def get_effect_by_type(self, effect_type: str) -> List:
        """Получение эффектов определенного типа"""
        return self.effect_categories.get(effect_type, [])
    
    def optimize_dataset_loading(self):
        """Оптимизация загрузки датасетов с использованием генераторов"""
        # Используем генераторы вместо загрузки всех данных в память
        def ave_generator():
            for scene_id, scene_data in self.ave_data.items():
                for shot_id, shot_info in scene_data.items():
                    yield {
                        'scene_id': scene_id,
                        'shot_id': shot_id,
                        'features': shot_info
                    }
        
        def edit3k_generator():
            for effect_id, effect_data in self.edit3k_data.items():
                effect_type = self.effect_types.get(effect_id)
                yield {
                    'effect_id': effect_id,
                    'effect_type': effect_type,
                    'data': effect_data
                }
        
        return ave_generator, edit3k_generator
    
    def create_tf_dataset(self, batch_size=32):
        """Создание TensorFlow Dataset для эффективного обучения"""
        logger.warning("TensorFlow не установлен. Метод create_tf_dataset недоступен в данной конфигурации.")
        return None
        
    def __del__(self):
        """Деструктор для остановки мониторинга памяти"""
        if self.use_optimization and hasattr(self, 'memory_manager'):
            self.memory_manager.stop_memory_monitoring() 