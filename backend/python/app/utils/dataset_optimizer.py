import os
import numpy as np
import json
import h5py
from typing import Dict, List, Tuple, Optional
import pickle
import shutil
from tqdm import tqdm
import logging

logger = logging.getLogger('DatasetOptimizer')

class DatasetOptimizer:
    """
    Утилита для оптимизации работы с датасетами AVE и Edit3K
    - Кэширование данных
    - Предобработка и сжатие датасетов
    - Оптимизация формата хранения
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def optimize_ave_dataset(self, ave_path: str) -> str:
        """
        Оптимизирует AVE датасет:
        - Извлекает только необходимые поля для анализа структуры
        - Сохраняет в более эффективном формате
        
        Args:
            ave_path: путь к оригинальному AVE датасету
            
        Returns:
            путь к оптимизированному датасету
        """
        print(f"Оптимизация AVE датасета из {ave_path}")
        ave_annotations_path = os.path.join(ave_path, 'annotations.json')
        
        if not os.path.exists(ave_annotations_path):
            raise FileNotFoundError(f"Файл аннотаций не найден: {ave_annotations_path}")
        
        # Путь для оптимизированного датасета
        optimized_path = os.path.join(self.cache_dir, 'ave_optimized.pkl')
        
        # Проверка существующего кэша
        if os.path.exists(optimized_path):
            print(f"Найден кэшированный AVE датасет: {optimized_path}")
            return optimized_path
        
        # Загрузка аннотаций
        with open(ave_annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Создаем упрощенную структуру для быстрого доступа
        optimized_data = {}
        
        for scene_id, scene_data in tqdm(annotations.items(), desc="Оптимизация AVE"):
            optimized_data[scene_id] = {}
            
            for shot_id, shot_data in scene_data.items():
                # Выбираем только необходимые поля
                optimized_shot = {
                    'shot-size': shot_data.get('shot-size', []),
                    'shot-angle': shot_data.get('shot-angle', []),
                    'shot-type': shot_data.get('shot-type', []),
                    'shot-motion': shot_data.get('shot-motion', [])
                }
                
                optimized_data[scene_id][shot_id] = optimized_shot
        
        # Сохранение оптимизированных данных
        with open(optimized_path, 'wb') as f:
            pickle.dump(optimized_data, f)
        
        print(f"AVE датасет оптимизирован и сохранен в {optimized_path}")
        return optimized_path
    
    def optimize_edit3k_dataset(self, edit3k_path: str) -> str:
        """
        Оптимизирует Edit3K датасет:
        - Преобразует JSON в HDF5 для эффективного доступа
        - Предобрабатывает кадры для быстрой загрузки
        
        Args:
            edit3k_path: путь к оригинальному Edit3K датасету
            
        Returns:
            путь к оптимизированному датасету
        """
        print(f"Оптимизация Edit3K датасета из {edit3k_path}")
        edit3k_json_path = os.path.join(edit3k_path, 'edit_3k.json')
        id2type_path = os.path.join(edit3k_path, 'edit_id2type.json')
        
        if not os.path.exists(edit3k_json_path) or not os.path.exists(id2type_path):
            raise FileNotFoundError(f"Файлы Edit3K не найдены")
        
        # Путь для оптимизированного датасета
        optimized_path = os.path.join(self.cache_dir, 'edit3k_optimized.h5')
        
        # Проверка существующего кэша
        if os.path.exists(optimized_path):
            print(f"Найден кэшированный Edit3K датасет: {optimized_path}")
            return optimized_path
        
        # Загрузка данных
        with open(edit3k_json_path, 'r') as f:
            edit3k_data = json.load(f)
            
        with open(id2type_path, 'r') as f:
            id2type = json.load(f)
        
        # Создаем HDF5 файл
        with h5py.File(optimized_path, 'w') as hf:
            # Сохраняем маппинг эффектов
            effect_types_group = hf.create_group('effect_types')
            for effect_id, effect_type in id2type.items():
                effect_types_group.attrs[effect_id] = effect_type
            
            # Группы по типам эффектов
            effects_group = hf.create_group('effects')
            
            # Обработка каждого эффекта
            for effect_id, effect_data in tqdm(edit3k_data.items(), desc="Оптимизация Edit3K"):
                effect_type = id2type.get(effect_id, 'unknown')
                
                # Создаем группу для текущего эффекта
                effect_group = effects_group.create_group(effect_id)
                effect_group.attrs['type'] = effect_type
                
                # Сохраняем метаданные эффекта
                for key, value in effect_data.items():
                    if key != 'frames':  # Кадры обрабатываем отдельно
                        if isinstance(value, (str, int, float, bool)):
                            effect_group.attrs[key] = value
                
                # Сохраняем кадры, если они есть
                if 'frames' in effect_data and effect_data['frames']:
                    frames = np.array(effect_data['frames'])
                    # Сжатие для экономии места
                    effect_group.create_dataset('frames', data=frames, compression='gzip')
        
        print(f"Edit3K датасет оптимизирован и сохранен в {optimized_path}")
        return optimized_path
    
    def load_optimized_ave(self, optimized_path: str) -> Dict:
        """Загрузка оптимизированного AVE датасета"""
        with open(optimized_path, 'rb') as f:
            return pickle.load(f)
    
    def load_optimized_edit3k(self, optimized_path: str) -> Tuple[Dict, Dict]:
        """
        Загрузка оптимизированного Edit3K датасета
        
        Returns:
            Tuple[эффекты, маппинг типов]
        """
        effects_data = {}
        effect_types = {}
        
        with h5py.File(optimized_path, 'r') as hf:
            # Загружаем маппинг типов
            for effect_id in hf['effect_types'].attrs:
                effect_types[effect_id] = hf['effect_types'].attrs[effect_id]
            
            # Загружаем данные эффектов по запросу
            for effect_id in hf['effects']:
                effect_group = hf['effects'][effect_id]
                effect_data = {}
                
                # Загружаем метаданные
                for key in effect_group.attrs:
                    effect_data[key] = effect_group.attrs[key]
                
                # Загружаем кадры, если они есть
                if 'frames' in effect_group:
                    effect_data['frames'] = effect_group['frames'][:]
                    
                effects_data[effect_id] = effect_data
                
        return effects_data, effect_types
    
    def clear_cache(self) -> None:
        """Очистка кэша оптимизированных датасетов"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Кэш очищен: {self.cache_dir}")
    
    @staticmethod
    def get_cached_frame_generator(frames_data):
        """Генератор для загрузки кадров по запросу"""
        for frame in frames_data:
            yield frame 