import tensorflow as tf
import numpy as np
import os
import json
import cv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

class EffectDetector:
    """
    Детектор эффектов и переходов в видео на основе датасета Edit3K.
    Модель обнаруживает различные эффекты: нарезки, растворения, затухания и т.д.
    """
    
    def __init__(self, model_path=None, input_shape=(10, 224, 224, 3)):
        """
        Инициализация детектора эффектов.
        
        Args:
            model_path: Путь к предобученной модели
            input_shape: Форма входных данных (sequence_length, height, width, channels)
        """
        self.input_shape = input_shape
        self.effect_categories = [
            "cut", "dissolve", "fade_in", "fade_out", "wipe", 
            "slide", "push", "flash", "cross_zoom", "rotation", 
            "blur_transition", "color_transfer"
        ]
        self.num_classes = len(self.effect_categories)
        
        # Создаем или загружаем модель
        if model_path and os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Инициализация новой модели EffectDetector")
            self.model = self._build_model()
    
    def _build_model(self):
        """Создает архитектуру модели на основе CNN-LSTM для обработки последовательности кадров"""
        # Базовая CNN модель для извлечения признаков из отдельных кадров
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape[1:]
        )
        
        # Замораживаем часть слоев базовой модели
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Создаем модель для извлечения признаков из отдельных кадров
        frame_model = Sequential([
            base_model,
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten()
        ])
        
        # Создаем последовательную модель для анализа изменений во времени
        sequence_model = Sequential([
            TimeDistributed(frame_model, input_shape=self.input_shape),
            LSTM(256, return_sequences=True),
            Dropout(0.5),
            LSTM(128),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Компилируем модель
        sequence_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return sequence_model
    
    def train(self, train_data, validation_data, epochs=50, batch_size=16, callbacks=None):
        """
        Обучает модель на предоставленных данных.
        
        Args:
            train_data: Обучающие данные в формате (x_train, y_train)
            validation_data: Валидационные данные в формате (x_val, y_val)
            epochs: Количество эпох обучения
            batch_size: Размер батча
            callbacks: Дополнительные callback-функции
        
        Returns:
            История обучения
        """
        if callbacks is None:
            callbacks = self._default_callbacks()
        
        x_train, y_train = train_data
        x_val, y_val = validation_data
        
        # Конвертируем метки в one-hot encoding
        y_train_onehot = to_categorical(y_train, self.num_classes)
        y_val_onehot = to_categorical(y_val, self.num_classes)
        
        history = self.model.fit(
            x_train, y_train_onehot,
            validation_data=(x_val, y_val_onehot),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _default_callbacks(self):
        """Создает стандартный набор callbacks для обучения"""
        checkpoint = ModelCheckpoint(
            'effect_detector_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [checkpoint, early_stopping, reduce_lr]
    
    def predict(self, frame_sequences):
        """
        Предсказывает эффекты для последовательности кадров.
        
        Args:
            frame_sequences: Массив последовательностей кадров формы (n, seq_len, height, width, channels)
            
        Returns:
            Предсказанные классы и вероятности
        """
        if len(frame_sequences.shape) == 4:  # Если передана одна последовательность
            frame_sequences = np.expand_dims(frame_sequences, axis=0)
        
        # Предобработка изображений
        preprocessed_sequences = self._preprocess_sequences(frame_sequences)
        
        # Получаем предсказания
        predictions = self.model.predict(preprocessed_sequences)
        
        # Получаем индексы классов с максимальной вероятностью
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Преобразуем индексы в названия классов
        predicted_labels = [self.effect_categories[idx] for idx in predicted_classes]
        
        # Вероятности предсказаний
        probabilities = np.max(predictions, axis=1)
        
        return predicted_labels, probabilities
    
    def _preprocess_sequences(self, sequences):
        """Предобработка последовательностей кадров для подачи в модель"""
        processed_sequences = []
        
        for sequence in sequences:
            processed_frames = []
            
            for frame in sequence:
                # Изменяем размер до входного размера модели
                resized_frame = cv2.resize(frame, (self.input_shape[1], self.input_shape[2]))
                
                # Нормализация пикселей
                normalized_frame = resized_frame / 255.0
                
                processed_frames.append(normalized_frame)
            
            processed_sequences.append(processed_frames)
        
        return np.array(processed_sequences)
    
    def extract_frame_sequences(self, video_path, effect_data, sequence_length=10):
        """
        Извлекает последовательности кадров для обнаруженных эффектов.
        
        Args:
            video_path: Путь к видеофайлу
            effect_data: Данные об эффектах в формате {'start': float, 'end': float, 'type': str}
            sequence_length: Количество кадров в последовательности
            
        Returns:
            Последовательности кадров и их метки
        """
        sequences = []
        labels = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for effect in effect_data:
            effect_type = effect['type']
            start_time = effect['start']
            end_time = effect['end']
            
            # Находим индекс категории эффекта
            if effect_type in self.effect_categories:
                label_idx = self.effect_categories.index(effect_type)
                
                # Вычисляем кадры для извлечения
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # Если длительность эффекта позволяет извлечь нужное количество кадров
                if end_frame - start_frame >= sequence_length:
                    # Извлекаем последовательность кадров
                    frames = []
                    step = (end_frame - start_frame) / sequence_length
                    
                    for i in range(sequence_length):
                        frame_idx = start_frame + int(i * step)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if ret:
                            frames.append(frame)
                        else:
                            break
                    
                    # Если удалось извлечь все кадры
                    if len(frames) == sequence_length:
                        sequences.append(frames)
                        labels.append(label_idx)
        
        cap.release()
        
        return np.array(sequences), np.array(labels)
    
    def load_edit3k_dataset(self, dataset_path, split_ratio=0.8):
        """
        Загружает и подготавливает датасет Edit3K для обучения.
        
        Args:
            dataset_path: Путь к датасету Edit3K
            split_ratio: Соотношение обучающей и валидационной выборок
            
        Returns:
            (x_train, y_train), (x_val, y_val)
        """
        # Загрузка метаданных
        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        samples = metadata.get('samples', [])
        
        sequences = []
        labels = []
        
        # Для каждого примера в датасете
        for sample in samples:
            filename = sample['filename']
            file_path = os.path.join(dataset_path, filename)
            
            # Если файл существует
            if os.path.exists(file_path):
                # Извлекаем последовательности кадров для эффектов
                effects = sample.get('effects', [])
                sample_sequences, sample_labels = self.extract_frame_sequences(
                    file_path, effects, self.input_shape[0]
                )
                
                sequences.extend(sample_sequences)
                labels.extend(sample_labels)
        
        # Конвертируем в массивы numpy
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # Перемешиваем данные
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        sequences = sequences[indices]
        labels = labels[indices]
        
        # Разделяем на обучающую и валидационную выборки
        x_train, x_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=1-split_ratio, random_state=42
        )
        
        return (x_train, y_train), (x_val, y_val)
    
    def save_model(self, path):
        """Сохраняет модель по указанному пути"""
        self.model.save(path)
        print(f"Модель сохранена по пути: {path}")


# Пример использования:
if __name__ == "__main__":
    # Инициализация детектора эффектов
    detector = EffectDetector()
    
    # Загрузка датасета Edit3K
    dataset_path = "datasets/Edit3K"
    (x_train, y_train), (x_val, y_val) = detector.load_edit3k_dataset(dataset_path)
    
    # Обучение модели
    detector.train((x_train, y_train), (x_val, y_val), epochs=30)
    
    # Сохранение модели
    detector.save_model("backend/ai_service/trained_models/effect_detector_model.h5") 