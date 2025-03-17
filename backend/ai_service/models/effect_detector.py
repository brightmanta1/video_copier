import tensorflow as tf
import numpy as np
import os
from typing import List, Dict, Any

class EffectDetector:
    """Класс для определения эффектов в видео с использованием TensorFlow"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.effect_types = []
        
        # Загрузка модели если указан путь
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Загрузка предобученной модели"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Загрузка маппинга эффектов если есть
            labels_path = os.path.join(os.path.dirname(model_path), "effect_labels.txt")
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as file:
                    self.effect_types = [line.strip() for line in file.readlines()]
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def train_model(self, dataset_manager, batch_size=32, epochs=10, validation_split=0.2) -> None:
        """Обучение модели на основе данных из Edit3K dataset"""
        # Получение данных из датасет-менеджера
        tf_dataset = dataset_manager.create_tf_dataset(batch_size=batch_size)
        
        # Создание простой CNN модели
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(dataset_manager.effect_categories), activation='softmax')
        ])
        
        # Компиляция модели
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callback для ранней остановки
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Обучение модели
        model.fit(
            tf_dataset,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        
        self.model = model
        self.effect_types = list(dataset_manager.effect_categories.keys())
    
    def detect_effects(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """
        Определение эффектов на кадрах
        
        Args:
            frames: массив кадров (N, H, W, 3)
            
        Returns:
            список эффектов с временными метками
        """
        if self.model is None:
            print("Model not loaded, cannot detect effects")
            return []
        
        # Масштабирование кадров до размера, ожидаемого моделью
        input_shape = self.model.input_shape[1:3]  # (height, width)
        scaled_frames = []
        
        for frame in frames:
            scaled = tf.image.resize(frame, input_shape)
            scaled_frames.append(scaled)
        
        # Нормализация
        scaled_frames = np.array(scaled_frames) / 255.0
        
        # Разбиваем на батчи для обработки
        batch_size = 32
        num_frames = len(scaled_frames)
        all_preds = []
        
        for i in range(0, num_frames, batch_size):
            batch = scaled_frames[i:i + batch_size]
            preds = self.model.predict(batch)
            all_preds.append(preds)
        
        # Объединяем предсказания
        all_preds = np.vstack(all_preds)
        
        # Формируем результаты
        effects = []
        current_effect = None
        effect_start = 0
        
        for i, pred in enumerate(all_preds):
            effect_idx = np.argmax(pred)
            confidence = pred[effect_idx]
            
            if confidence > 0.5:  # Порог уверенности
                effect_type = self.effect_types[effect_idx] if effect_idx < len(self.effect_types) else f"effect_{effect_idx}"
                
                # Проверяем, продолжается ли текущий эффект
                if current_effect == effect_type:
                    continue
                
                # Если ранее был эффект, добавляем его в список
                if current_effect:
                    effects.append({
                        'type': current_effect,
                        'start_frame': effect_start,
                        'end_frame': i - 1,
                        'confidence': float(confidence)
                    })
                
                # Начинаем новый эффект
                current_effect = effect_type
                effect_start = i
        
        # Добавляем последний эффект, если он был
        if current_effect:
            effects.append({
                'type': current_effect,
                'start_frame': effect_start,
                'end_frame': len(all_preds) - 1,
                'confidence': float(confidence)
            })
            
        return effects
    
    def save_model(self, model_dir: str) -> None:
        """Сохранение модели"""
        if not self.model:
            print("No model to save")
            return
            
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "effect_detection_model")
        self.model.save(model_path)
        
        # Сохраняем маппинг эффектов
        labels_path = os.path.join(model_dir, "effect_labels.txt")
        with open(labels_path, 'w') as file:
            for effect_type in self.effect_types:
                file.write(f"{effect_type}\n")
                
        print(f"Model saved to {model_path}")
        print(f"Labels saved to {labels_path}") 