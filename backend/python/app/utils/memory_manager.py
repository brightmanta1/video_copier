import os
import psutil
import gc
import numpy as np
from typing import List, Any, Callable, Optional
import time
import threading
import logging

logger = logging.getLogger('MemoryManager')

class MemoryManager:
    """
    Менеджер памяти для оптимизации обработки больших объемов данных:
    - Отслеживание использования памяти
    - Автоматическая сборка мусора при достижении порогов
    - Оптимизация загрузки больших массивов видеоданных
    """
    
    def __init__(self, memory_limit_percent: float = 80.0, check_interval: float = 5.0):
        """
        Args:
            memory_limit_percent: пороговый процент использования памяти
            check_interval: интервал проверки использования памяти (сек)
        """
        self.memory_limit_percent = memory_limit_percent
        self.check_interval = check_interval
        self.monitoring_active = False
        self.monitor_thread = None
    
    def get_memory_usage(self) -> float:
        """
        Получение текущего использования памяти в процентах
        
        Returns:
            процент использования памяти
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        total_memory = psutil.virtual_memory().total
        
        # Использование в процентах
        usage_percent = (mem_info.rss / total_memory) * 100
        return usage_percent
    
    def batch_process(self, items: List[Any], processor: Callable, batch_size: int = 32) -> List[Any]:
        """
        Обработка списка элементов батчами с контролем памяти
        
        Args:
            items: список элементов для обработки
            processor: функция обработки элемента
            batch_size: размер батча
            
        Returns:
            список результатов обработки
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            # Проверка использования памяти
            if self.get_memory_usage() > self.memory_limit_percent:
                logger.warning(f"Память превышает порог ({self.memory_limit_percent}%), запуск сборки мусора")
                self.force_garbage_collection()
            
            # Обработка батча
            batch = items[i:i + batch_size]
            batch_results = [processor(item) for item in batch]
            results.extend(batch_results)
        
        return results
    
    def process_video_frames(self, frames: np.ndarray, processor: Callable, batch_size: int = 32) -> np.ndarray:
        """
        Оптимизированная обработка кадров видео
        
        Args:
            frames: массив кадров
            processor: функция обработки кадра
            batch_size: размер батча
            
        Returns:
            массив обработанных кадров
        """
        num_frames = len(frames)
        processed_frames = []
        
        for i in range(0, num_frames, batch_size):
            # Проверка использования памяти
            if self.get_memory_usage() > self.memory_limit_percent:
                logger.warning(f"Память превышает порог при обработке кадров, запуск сборки мусора")
                self.force_garbage_collection()
                # Уменьшаем размер батча при недостатке памяти
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"Размер батча уменьшен до {batch_size}")
            
            # Обработка батча кадров
            batch = frames[i:i + batch_size]
            batch_processed = np.array([processor(frame) for frame in batch])
            processed_frames.append(batch_processed)
        
        # Объединение результатов
        if len(processed_frames) == 1:
            return processed_frames[0]
        return np.vstack(processed_frames)
    
    def start_memory_monitoring(self) -> None:
        """Запуск фонового мониторинга использования памяти"""
        if self.monitoring_active:
            logger.warning("Мониторинг памяти уже запущен")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_worker, daemon=True)
        self.monitor_thread.start()
        logger.info("Мониторинг памяти запущен")
    
    def stop_memory_monitoring(self) -> None:
        """Остановка фонового мониторинга использования памяти"""
        if not self.monitoring_active:
            logger.warning("Мониторинг памяти не запущен")
            return
            
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.info("Мониторинг памяти остановлен")
    
    def _memory_monitor_worker(self) -> None:
        """Фоновый поток для мониторинга использования памяти"""
        while self.monitoring_active:
            mem_usage = self.get_memory_usage()
            
            if mem_usage > self.memory_limit_percent:
                logger.warning(f"Использование памяти превышает порог: {mem_usage:.2f}%")
                self.force_garbage_collection()
            
            time.sleep(self.check_interval)
    
    def force_garbage_collection(self) -> None:
        """Принудительная сборка мусора для освобождения памяти"""
        # Сохраняем значение использования памяти до сборки мусора
        before = self.get_memory_usage()
        
        # Запускаем сборку мусора
        gc.collect()
        
        # Сохраняем значение использования памяти после сборки мусора
        after = self.get_memory_usage()
        
        # Логируем результат
        logger.info(f"Сборка мусора: {before:.2f}% -> {after:.2f}% (освобождено {before - after:.2f}%)")
    
    @staticmethod
    def get_optimal_batch_size(frame_shape: tuple, available_memory_mb: Optional[int] = None) -> int:
        """
        Рассчитывает оптимальный размер батча на основе размера кадра и доступной памяти
        
        Args:
            frame_shape: размеры кадра (высота, ширина, каналы)
            available_memory_mb: доступная память в МБ (опционально)
            
        Returns:
            оптимальный размер батча
        """
        # Если не указана доступная память, используем 50% от свободной системной памяти
        if available_memory_mb is None:
            free_memory = psutil.virtual_memory().available
            available_memory_mb = int((free_memory / (1024 * 1024)) * 0.5)
        
        # Рассчитываем размер одного кадра в байтах
        # Предполагаем, что кадр имеет тип float32 (4 байта)
        frame_size_bytes = np.prod(frame_shape) * 4
        
        # Рассчитываем, сколько кадров поместится в доступную память
        # Учитываем накладные расходы, поэтому используем 80% от расчетного количества
        max_frames = int((available_memory_mb * 1024 * 1024 * 0.8) / frame_size_bytes)
        
        # Выбираем размер батча как степень двойки, не превышающую max_frames
        batch_size = 1
        while batch_size * 2 <= max_frames and batch_size < 128:
            batch_size *= 2
            
        return max(1, min(batch_size, 128))  # Не больше 128 кадров в батче 