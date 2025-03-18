import os
import numpy as np
import logging
from pathlib import Path

# Настраиваем логирование
logger = logging.getLogger('example_model')

# Импортируем модули AI сервиса
from backend.ai_service.models import load_model, MODELS_DIR, TENSORFLOW_AVAILABLE

class ExampleModel:
    """
    Пример класса модели, который использует TensorFlow через Docker при необходимости
    """
    
    def __init__(self):
        self.model = None
        self.model_path = MODELS_DIR / "example_model.h5"
        self.initialized = False
        
    def initialize(self, force_reload=False):
        """
        Инициализирует модель
        
        Args:
            force_reload: принудительно перезагрузить модель
            
        Returns:
            True, если инициализация успешна, иначе False
        """
        if self.initialized and not force_reload:
            return True
            
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow недоступен, инициализация модели невозможна")
            return False
            
        # Создаем простую модель, если файла модели нет
        if not os.path.exists(self.model_path):
            logger.info(f"Создание новой модели, так как файл не существует: {self.model_path}")
            self._create_model()
        else:
            # Загружаем существующую модель
            logger.info(f"Загрузка существующей модели: {self.model_path}")
            self.model = load_model(self.model_path)
            
        self.initialized = self.model is not None
        return self.initialized
    
    def _create_model(self):
        """Создает простую модель для примера"""
        try:
            # Импортируем tensorflow через общий интерфейс
            from backend.ai_service.models import tf
            
            # Создаем простую модель для тестирования
            inputs = tf.keras.Input(shape=(100,), name='input')
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            outputs = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
            
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Сохраняем модель
            self.model.save(self.model_path)
            logger.info(f"Новая модель создана и сохранена в: {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка при создании модели: {str(e)}")
            self.model = None
    
    def predict(self, data):
        """
        Выполняет предсказание на основе входных данных
        
        Args:
            data: входные данные для предсказания
            
        Returns:
            результат предсказания или None в случае ошибки
        """
        if not self.initialized:
            initialized = self.initialize()
            if not initialized:
                logger.error("Не удалось инициализировать модель")
                return None
                
        try:
            # Преобразуем данные в нужный формат, если необходимо
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                
            # Проверяем формат входных данных
            if data.ndim == 1:
                data = np.expand_dims(data, 0)  # Добавляем размерность батча
                
            # Нормализуем данные, если необходимо
            # ...
                
            # Выполняем предсказание
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
            return None
            
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Обучает модель на предоставленных данных
        
        Args:
            x_train: обучающие данные
            y_train: обучающие метки
            x_val: валидационные данные
            y_val: валидационные метки
            epochs: количество эпох обучения
            batch_size: размер батча
            
        Returns:
            история обучения или None в случае ошибки
        """
        if not self.initialized:
            initialized = self.initialize()
            if not initialized:
                logger.error("Не удалось инициализировать модель")
                return None
                
        try:
            # Обучаем модель
            history = self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Сохраняем обученную модель
            self.model.save(self.model_path)
            logger.info(f"Модель обучена и сохранена в: {self.model_path}")
            
            return history.history
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            return None
            

# Создаем экземпляр класса для использования
example_model = ExampleModel() 