import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger('CacheManager')

class CacheManager:
    """
    Менеджер кеша для хранения результатов анализа видео и промежуточных данных.
    Позволяет значительно ускорить повторную обработку видео.
    """
    
    def __init__(self, cache_dir: str = 'cache/results', max_age_days: int = 30, 
                 enabled: bool = True, max_size_mb: int = 5000):
        """
        Args:
            cache_dir: директория для хранения кеша
            max_age_days: максимальный возраст кеша в днях
            enabled: включен ли кеш
            max_size_mb: максимальный размер кеша в МБ
        """
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        self.enabled = enabled
        self.max_size_mb = max_size_mb
        
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)
            # Запускаем очистку устаревшего кеша при инициализации
            self._cleanup_old_cache()
    
    def cache_result(self, key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Кеширует результат обработки
        
        Args:
            key: уникальный ключ (хеш видео или параметров)
            data: данные для кеширования
            metadata: дополнительные метаданные
            
        Returns:
            путь к файлу кеша
        """
        if not self.enabled:
            return ""
            
        # Создаем хеш ключа для имени файла
        cache_key = self._compute_hash(key)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Добавляем метаданные о времени создания кеша
        if metadata is None:
            metadata = {}
        metadata['created_at'] = time.time()
        metadata['source_key'] = key
        
        # Структура для хранения
        cache_data = {
            'data': data,
            'metadata': metadata
        }
        
        # Сохраняем кеш
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кеш сохранен: {cache_path}")
            
            # Проверяем общий размер кеша
            self._check_cache_size()
            
            return cache_path
        except Exception as e:
            logger.error(f"Ошибка при сохранении кеша: {e}")
            return ""
    
    def get_cached_result(self, key: str) -> Optional[Dict]:
        """
        Получает результат из кеша
        
        Args:
            key: уникальный ключ
            
        Returns:
            данные из кеша или None, если кеш не найден или устарел
        """
        if not self.enabled:
            return None
            
        cache_key = self._compute_hash(key)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Проверяем возраст кеша
            created_at = cache_data.get('metadata', {}).get('created_at', 0)
            age_days = (time.time() - created_at) / (60 * 60 * 24)
            
            if age_days > self.max_age_days:
                logger.info(f"Кеш устарел (возраст: {age_days:.1f} дней): {cache_path}")
                os.remove(cache_path)
                return None
                
            logger.info(f"Получен кеш: {cache_path}")
            return cache_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке кеша: {e}")
            return None
    
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """
        Инвалидирует кеш по ключу или весь кеш
        
        Args:
            key: уникальный ключ или None для очистки всего кеша
        """
        if not self.enabled:
            return
            
        if key is None:
            # Очищаем весь кеш
            self._clear_all_cache()
        else:
            # Очищаем конкретный кеш
            cache_key = self._compute_hash(key)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Кеш инвалидирован: {cache_path}")
    
    def cached_execution(self, function: Callable, *args, **kwargs) -> Any:
        """
        Декоратор для кеширования результатов выполнения функции
        
        Args:
            function: функция для выполнения
            args, kwargs: аргументы функции
            
        Returns:
            результат выполнения функции (из кеша или непосредственно)
        """
        if not self.enabled:
            return function(*args, **kwargs)
            
        # Создаем ключ на основе имени функции и аргументов
        key_parts = [function.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
        cache_key = "_".join(key_parts)
        
        # Проверяем кеш
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Возвращен результат из кеша для: {function.__name__}")
            return cached_result['data']
            
        # Выполняем функцию и кешируем результат
        result = function(*args, **kwargs)
        self.cache_result(cache_key, result, {
            'function': function.__name__,
            'args': str(args),
            'kwargs': str(kwargs)
        })
        
        return result
    
    def _compute_hash(self, key: str) -> str:
        """Вычисляет хеш для ключа"""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _cleanup_old_cache(self) -> None:
        """Очищает устаревший кеш"""
        if not os.path.exists(self.cache_dir):
            return
            
        current_time = time.time()
        max_age_seconds = self.max_age_days * 24 * 60 * 60
        
        for file_name in os.listdir(self.cache_dir):
            if not file_name.endswith('.pkl'):
                continue
                
            file_path = os.path.join(self.cache_dir, file_name)
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                os.remove(file_path)
                logger.info(f"Удален устаревший кеш: {file_path}")
    
    def _check_cache_size(self) -> None:
        """Проверяет общий размер кеша и при необходимости удаляет старые файлы"""
        if not os.path.exists(self.cache_dir):
            return
            
        # Получаем список файлов с их временем создания
        files = []
        total_size_mb = 0
        
        for file_name in os.listdir(self.cache_dir):
            if not file_name.endswith('.pkl'):
                continue
                
            file_path = os.path.join(self.cache_dir, file_name)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_mb += size_mb
            
            files.append({
                'path': file_path,
                'mtime': os.path.getmtime(file_path),
                'size_mb': size_mb
            })
        
        # Если размер кеша превышает лимит, удаляем старые файлы
        if total_size_mb > self.max_size_mb:
            # Сортируем файлы по времени создания (от старых к новым)
            files.sort(key=lambda x: x['mtime'])
            
            # Удаляем старые файлы, пока размер не станет меньше лимита
            for file_info in files:
                if total_size_mb <= self.max_size_mb:
                    break
                    
                os.remove(file_info['path'])
                total_size_mb -= file_info['size_mb']
                logger.info(f"Удален старый кеш для освобождения места: {file_info['path']}")
    
    def _clear_all_cache(self) -> None:
        """Очищает весь кеш"""
        if not os.path.exists(self.cache_dir):
            return
            
        for file_name in os.listdir(self.cache_dir):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, file_name)
                os.remove(file_path)
        
        logger.info(f"Весь кеш очищен: {self.cache_dir}")
        
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        logger.info("CacheManager уничтожен") 