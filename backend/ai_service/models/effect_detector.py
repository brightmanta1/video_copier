import os
import json
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

# Настраиваем логирование
logger = logging.getLogger('effect_detector')

# Импортируем модули AI сервиса
from backend.ai_service.models import load_model, MODELS_DIR, TENSORFLOW_AVAILABLE

# Если TensorFlow доступен, импортируем его
if TENSORFLOW_AVAILABLE:
    from backend.ai_service.models import tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, TimeDistributed, BatchNormalization
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical

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
        self.initialized = False
        
        # Устанавливаем путь к модели
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = MODELS_DIR / "effect_detector.h5"
        
        # Инициализируем модель
        self.initialize()
    
    def initialize(self, force_reload=False):
        """
        Инициализирует модель детектора эффектов
        
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
            
        # Создаем или загружаем модель
        if os.path.exists(self.model_path):
            logger.info(f"Загрузка модели из {self.model_path}")
            self.model = load_model(str(self.model_path))
        else:
            logger.info("Инициализация новой модели EffectDetector")
            self.model = self._build_model()
            
        self.initialized = self.model is not None
        return self.initialized
    
    def _build_model(self):
        """Создает архитектуру модели на основе CNN-LSTM для обработки последовательности кадров"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow недоступен, создание модели невозможно")
            return None
            
        try:
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
        except Exception as e:
            logger.error(f"Ошибка при создании модели: {str(e)}")
            return None
    
    def train(self, train_data, validation_data=None, epochs=50, batch_size=16, callbacks=None):
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
        if not self.initialized:
            initialized = self.initialize()
            if not initialized:
                logger.error("Не удалось инициализировать модель")
                return None
                
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow недоступен, обучение модели невозможно")
            return None
            
        try:
            if callbacks is None:
                callbacks = self._default_callbacks()
            
            x_train, y_train = train_data
            
            # Если validation_data не предоставлены, разделяем обучающие данные
            if validation_data is None:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, test_size=0.2, random_state=42
                )
                validation_data = (x_val, y_val)
            else:
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
            
            # Сохраняем модель
            self.save_model(self.model_path)
            
            return history.history
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            return None
    
    def _default_callbacks(self):
        """Создает стандартный набор callbacks для обучения"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow недоступен, создание callbacks невозможно")
            return []
            
        checkpoint = ModelCheckpoint(
            str(self.model_path),
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
        if not self.initialized:
            initialized = self.initialize()
            if not initialized:
                logger.error("Не удалось инициализировать модель")
                return None, None
                
        try:
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
        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
            return None, None
    
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
        
        try:
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
        except Exception as e:
            logger.error(f"Ошибка при извлечении последовательностей кадров: {str(e)}")
            return np.array([]), np.array([])
    
    def load_edit3k_dataset(self, dataset_path, split_ratio=0.8):
        """
        Загружает и подготавливает данные из датасета Edit3K.
        
        Args:
            dataset_path: Путь к директории с датасетом
            split_ratio: Соотношение обучающей/валидационной выборки
            
        Returns:
            Кортеж (x_train, y_train, x_val, y_val)
        """
        try:
            dataset_path = Path(dataset_path)
            
            # Загружаем метаданные о эффектах из JSON
            metadata_file = dataset_path / "effects_metadata.json"
            if not metadata_file.exists():
                logger.error(f"Файл метаданных не найден: {metadata_file}")
                return None, None, None, None
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            sequences = []
            labels = []
            
            # Для каждого видео в метаданных
            for video_id, effects in metadata.items():
                video_path = dataset_path / "videos" / f"{video_id}.mp4"
                
                if not video_path.exists():
                    logger.warning(f"Видеофайл не найден: {video_path}")
                    continue
                    
                # Извлекаем последовательности кадров для каждого эффекта
                video_sequences, video_labels = self.extract_frame_sequences(
                    str(video_path), effects
                )
                
                if len(video_sequences) > 0:
                    sequences.extend(video_sequences)
                    labels.extend(video_labels)
            
            sequences = np.array(sequences)
            labels = np.array(labels)
            
            # Делим данные на обучающую и валидационную выборки
            x_train, x_val, y_train, y_val = train_test_split(
                sequences, labels, train_size=split_ratio, random_state=42
            )
            
            logger.info(f"Загружено {len(sequences)} последовательностей, {len(set(labels))} классов")
            logger.info(f"Размер обучающей выборки: {len(x_train)}, валидационной: {len(x_val)}")
            
            return x_train, y_train, x_val, y_val
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета Edit3K: {str(e)}")
            return None, None, None, None
    
    def save_model(self, path):
        """Сохраняет модель по указанному пути"""
        if not self.initialized:
            logger.error("Модель не инициализирована, сохранение невозможно")
            return False
            
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохраняем модель
            self.model.save(path)
            logger.info(f"Модель успешно сохранена: {path}")
            
            # Сохраняем категории эффектов
            categories_path = os.path.join(os.path.dirname(path), "effect_categories.json")
            with open(categories_path, 'w') as f:
                json.dump(self.effect_categories, f)
                
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {str(e)}")
            return False


# Создаем экземпляр класса для использования
effect_detector = EffectDetector()

# Пример использования:
if __name__ == "__main__":
    # Загрузка датасета Edit3K
    dataset_path = "datasets/Edit3K"
    (x_train, y_train, x_val, y_val) = effect_detector.load_edit3k_dataset(dataset_path)
    
    # Обучение модели
    history = effect_detector.train((x_train, y_train), (x_val, y_val), epochs=30)
    
    # Сохранение модели
    effect_detector.save_model("backend/ai_service/trained_models/effect_detector_model.h5") 