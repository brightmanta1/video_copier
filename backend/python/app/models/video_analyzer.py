import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

# Импортируем утилиты из нашего приложения
from ..utils.memory_manager import MemoryManager
from ..utils.cache_manager import CacheManager

logger = logging.getLogger('VideoAnalyzer')

class VideoAnalyzer:
    def __init__(self, dataset_manager=None, shot_threshold=30, use_memory_optimization=True, use_cache=True):
        self.shot_threshold = shot_threshold
        self.effect_model = None
        self.batch_size = 32
        self.dataset_manager = dataset_manager
        
        # Инициализация оптимизатора памяти
        if use_memory_optimization:
            try:
                self.memory_manager = MemoryManager()
                self.memory_manager.start_memory_monitoring()
            except Exception as e:
                logger.warning(f"Невозможно инициализировать оптимизатор памяти: {e}")
                self.memory_manager = None
        else:
            self.memory_manager = None
            
        # Инициализация менеджера кеша
        if use_cache:
            try:
                self.cache_manager = CacheManager()
            except Exception as e:
                logger.warning(f"Невозможно инициализировать менеджер кеша: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None
        
    def extract_frames(self, video_path, sample_rate=1):
        """
        Извлекает кадры из видео с заданной частотой
        Args:
            video_path: путь к видео файлу
            sample_rate: частота извлечения кадров (1 = каждый кадр)
        Returns:
            frames: список numpy массивов кадров
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # Оптимизация использования памяти
        if self.memory_manager:
            # Получаем информацию о размере кадра для определения оптимального размера батча
            ret, first_frame = cap.read()
            if not ret:
                return np.array([])  # Возвращаем пустой массив, если видео не открывается
            
            # Оценка оптимального размера батча
            if first_frame is not None:
                frame_shape = first_frame.shape
                self.batch_size = self.memory_manager.get_optimal_batch_size(frame_shape)
                frames.append(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            
            current_batch = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    current_batch.append(frame_rgb)
                    
                    # Когда батч заполнен, обрабатываем его
                    if len(current_batch) >= self.batch_size:
                        # Проверка памяти
                        if self.memory_manager.get_memory_usage() > 80:
                            self.memory_manager.force_garbage_collection()
                            
                        # Добавляем пакет кадров
                        frames.extend(current_batch)
                        current_batch = []
            
            # Добавляем оставшиеся кадры
            if current_batch:
                frames.extend(current_batch)
        else:
            # Обычное извлечение кадров
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
            
        cap.release()
        return np.array(frames)
    
    def detect_shot_boundaries(self, frames):
        """
        Определяет границы сцен в видео
        Args:
            frames: список кадров
        Returns:
            boundaries: список индексов кадров, где происходит смена сцены
        """
        boundaries = [0]  # Начинаем с первого кадра
        prev_frame = frames[0]
        
        # При наличии оптимизатора памяти обрабатываем большие видео по частям
        if self.memory_manager and len(frames) > 1000:  # Если больше 1000 кадров
            batch_size = min(1000, self.batch_size * 4)  # Увеличиваем размер батча для ускорения
            
            for i in range(1, len(frames), batch_size):
                end_idx = min(i + batch_size, len(frames))
                batch_frames = frames[i:end_idx]
                
                for j, frame in enumerate(batch_frames):
                    # Вычисляем разницу между текущим и предыдущим кадром
                    diff = np.mean(np.abs(frame - prev_frame))
                    
                    if diff > self.shot_threshold:
                        boundaries.append(i + j)
                        
                    prev_frame = frame
                
                # Очистка памяти между батчами
                if self.memory_manager and self.memory_manager.get_memory_usage() > 80:
                    self.memory_manager.force_garbage_collection()
        else:
            # Обычное обнаружение границ сцен
            for i in range(1, len(frames)):
                # Вычисляем разницу между текущим и предыдущим кадром
                diff = np.mean(np.abs(frames[i] - prev_frame))
                
                if diff > self.shot_threshold:
                    boundaries.append(i)
                    
                prev_frame = frames[i]
            
        # Добавляем последний кадр
        boundaries.append(len(frames))
            
        return boundaries
    
    def analyze_shot_type(self, frame):
        """
        Анализирует тип съемки используя модель из AVE dataset
        Args:
            frame: кадр для анализа
        Returns:
            shot_info: словарь с информацией о типе съемки
        """
        # Проверяем наличие датасет-менеджера
        if self.dataset_manager and hasattr(self.dataset_manager, 'shot_features'):
            # Заглушка для типа съемки с использованием статистики из AVE
            sizes = list(self.dataset_manager.shot_features['sizes'])
            angles = list(self.dataset_manager.shot_features['angles'])
            types = list(self.dataset_manager.shot_features['types'])
            motions = list(self.dataset_manager.shot_features['motions'])
            
            # Выбираем случайные значения из имеющихся в датасете
            # В реальной системе здесь будет классификация кадра
            return {
                'shot_size': sizes[0] if sizes else 'medium',
                'shot_angle': angles[0] if angles else 'eye-level',
                'shot_type': types[0] if types else 'single-shot',
                'shot_motion': motions[0] if motions else 'static'
            }
        
        # Если нет датасет-менеджера, используем значения по умолчанию
        return {
            'shot_size': 'medium',
            'shot_angle': 'eye-level',
            'shot_type': 'single-shot',
            'shot_motion': 'static'
        }

    def analyze_video(self, video_path, use_gpu=False):
        """
        Полный анализ видео
        Args:
            video_path: путь к видео файлу
            use_gpu: использовать ли GPU для анализа (в текущей версии не используется)
        Returns:
            analysis: словарь с полным анализом видео
        """
        # Проверяем наличие результатов в кеше
        if self.cache_manager:
            # Вычисляем уникальный ключ для видео на основе пути и времени изменения
            if os.path.exists(video_path):
                video_modified_time = os.path.getmtime(video_path)
                cache_key = f"{video_path}_{video_modified_time}_{self.shot_threshold}"
                
                cached_result = self.cache_manager.get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Используем кешированный результат анализа для {video_path}")
                    return cached_result['data']
        
        # Выполняем анализ
        analysis = self._analyze(video_path)
        
        # Сохраняем результат в кеш
        if self.cache_manager and os.path.exists(video_path):
            video_modified_time = os.path.getmtime(video_path)
            cache_key = f"{video_path}_{video_modified_time}_{self.shot_threshold}"
            
            self.cache_manager.cache_result(cache_key, analysis, {
                'video_path': video_path,
                'shot_threshold': self.shot_threshold
            })
            
        return analysis
    
    def _analyze(self, video_path):
        frames = self.extract_frames(video_path)
        
        # Оптимизируем использование памяти
        if self.memory_manager:
            frames = self.optimize_memory_usage(frames)
        
        boundaries = self.detect_shot_boundaries(frames)
        
        analysis = {
            'total_frames': len(frames),
            'shot_boundaries': boundaries,
            'shots': []
        }
        
        # Анализ каждой сцены
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Берем средний кадр из сцены для анализа
            middle_frame_idx = start + (end - start) // 2
            middle_frame_idx = min(middle_frame_idx, len(frames) - 1)
            
            shot_info = {
                'start_frame': start,
                'end_frame': end,
                'duration_frames': end - start,
                'features': self.analyze_shot_type(frames[middle_frame_idx])
            }
            
            analysis['shots'].append(shot_info)
            
        # Детекция эффектов
        if self.effect_model:
            # Если много кадров, используем батчевую обработку
            if self.memory_manager and len(frames) > 100:
                analysis['effects'] = self.detect_effects_batched(frames)
            else:
                # Для небольших видео можно обработать все сразу
                analysis['effects'] = self.detect_effects_batched(frames)
        else:
            analysis['effects'] = []
            
        return analysis
    
    def detect_effects_batched(self, frames):
        """Батчевое определение эффектов"""
        if not self.effect_model:
            return []
            
        effects = []
        
        # Если есть memory_manager, используем его для батчевой обработки
        if self.memory_manager:
            def process_batch(batch):
                # Здесь будет предсказание от модели
                # Заглушка
                return []
                
            # Определяем оптимальный размер батча
            batch_size = self.memory_manager.get_optimal_batch_size(frames[0].shape)
            
            # Обрабатываем кадры батчами
            for i in range(0, len(frames), batch_size):
                # Проверка использования памяти
                if self.memory_manager.get_memory_usage() > 80:
                    self.memory_manager.force_garbage_collection()
                    
                # Обработка батча
                batch_frames = frames[i:i + batch_size]
                batch_effects = process_batch(batch_frames)
                effects.extend(batch_effects)
        else:
            # Обычная обработка батчами
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                # Здесь будет предсказание от модели
                
        return effects
    
    def validate_predictions(self, predictions, ground_truth):
        """Валидация предсказаний"""
        metrics = {
            'accuracy': 0,
            'precision': {},
            'recall': {},
            'confusion_matrix': {}
        }
        
        # Подсчет метрик
        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            if pred['type'] == gt['type']:
                correct += 1
                
        # Общая точность
        metrics['accuracy'] = correct / len(predictions) if predictions else 0
        
        return metrics

    def optimize_memory_usage(self, frames):
        """Оптимизация использования памяти при обработке"""
        # Уменьшаем разрешение, если нужно
        if frames[0].shape[0] > 720:  # Если высота кадра больше 720
            # Если есть менеджер памяти, используем его для оптимизированной обработки
            if self.memory_manager:
                def resize_frame(frame):
                    return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                return self.memory_manager.process_video_frames(frames, resize_frame, batch_size=self.batch_size)
            
            # Обычное масштабирование
            scaled_frames = []
            for frame in frames:
                scaled = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                scaled_frames.append(scaled)
            return np.array(scaled_frames)
        
        return frames
        
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        # Останавливаем мониторинг памяти при уничтожении объекта
        if hasattr(self, 'memory_manager') and self.memory_manager:
            self.memory_manager.stop_memory_monitoring() 