import numpy as np
import os
import json
import cv2
from backend.python.app.utils.tensorflow_docker import import_tensorflow

# Импортируем TensorFlow через Docker
tf = import_tensorflow()
if tf is None:
    raise ImportError("TensorFlow через Docker недоступен")

class ShotClassifier:
    """
    Классификатор типов кадров (shot types) на основе датасета AVE.
    Модель обучается определять типы кадров: широкий план, средний план, крупный план и т.д.
    """
    
    def __init__(self, model_path=None, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.shot_categories = [
            "wide_shot", "medium_shot", "close_up", "tracking_shot", 
            "pan", "tilt", "zoom", "static", "slow_motion", "timelapse", "action"
        ]
        self.num_classes = len(self.shot_categories)
        
        # Создаем или загружаем модель
        if model_path and os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Инициализация новой модели ShotClassifier")
            self.model = self._build_model()
    
    def _build_model(self):
        """Создает архитектуру модели на основе ResNet50"""
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Замораживаем часть слоев базовой модели
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32, callbacks=None):
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
        y_train_onehot = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
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
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'shot_classifier_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        return [checkpoint, early_stopping, reduce_lr]
    
    def predict(self, frames):
        """
        Предсказывает типы кадров для набора фреймов.
        
        Args:
            frames: Массив фреймов формы (n, height, width, channels)
            
        Returns:
            Предсказанные классы и вероятности
        """
        if len(frames.shape) == 3:  # Если передан один кадр
            frames = np.expand_dims(frames, axis=0)
        
        # Предобработка изображений
        preprocessed_frames = self._preprocess_frames(frames)
        
        # Получаем предсказания
        predictions = self.model.predict(preprocessed_frames)
        
        # Получаем индексы классов с максимальной вероятностью
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Преобразуем индексы в названия классов
        predicted_labels = [self.shot_categories[idx] for idx in predicted_classes]
        
        # Вероятности предсказаний
        probabilities = np.max(predictions, axis=1)
        
        return predicted_labels, probabilities
    
    def _preprocess_frames(self, frames):
        """Предобработка фреймов для подачи в модель"""
        processed_frames = []
        
        for frame in frames:
            # Изменяем размер до входного размера модели
            resized_frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
            
            # Нормализация пикселей
            normalized_frame = resized_frame / 255.0
            
            processed_frames.append(normalized_frame)
        
        return np.array(processed_frames)
    
    def load_ave_dataset(self, dataset_path, split_ratio=0.8):
        """
        Загружает и подготавливает датасет AVE для обучения.
        
        Args:
            dataset_path: Путь к датасету AVE
            split_ratio: Соотношение обучающей и валидационной выборок
            
        Returns:
            (x_train, y_train), (x_val, y_val)
        """
        # Загрузка метаданных
        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        samples = metadata.get('samples', [])
        categories = metadata.get('categories', [])
        
        frames = []
        labels = []
        
        # Для каждого примера в датасете
        for sample in samples:
            filename = sample['filename']
            file_path = os.path.join(dataset_path, filename)
            
            # Если файл существует
            if os.path.exists(file_path):
                # Открываем видеофайл
                cap = cv2.VideoCapture(file_path)
                
                # Получаем аннотации
                annotations = sample.get('annotations', [])
                
                for annotation in annotations:
                    category = annotation['category']
                    start_time = annotation['start']
                    end_time = annotation['end']
                    
                    # Находим индекс категории
                    if category in categories:
                        label_idx = categories.index(category)
                        
                        # Получаем кадр из середины аннотированного отрезка
                        middle_time = (start_time + end_time) / 2
                        cap.set(cv2.CAP_PROP_POS_MSEC, middle_time * 1000)
                        ret, frame = cap.read()
                        
                        if ret:
                            frames.append(frame)
                            labels.append(label_idx)
                
                cap.release()
        
        # Конвертируем в массивы numpy
        frames = np.array(frames)
        labels = np.array(labels)
        
        # Перемешиваем данные
        indices = np.arange(len(frames))
        np.random.shuffle(indices)
        frames = frames[indices]
        labels = labels[indices]
        
        # Разделяем на обучающую и валидационную выборки
        split_idx = int(len(frames) * split_ratio)
        x_train, y_train = frames[:split_idx], labels[:split_idx]
        x_val, y_val = frames[split_idx:], labels[split_idx:]
        
        return (x_train, y_train), (x_val, y_val)
    
    def save_model(self, path):
        """Сохраняет модель по указанному пути"""
        self.model.save(path)
        print(f"Модель сохранена по пути: {path}")

# Пример использования:
if __name__ == "__main__":
    # Инициализация классификатора
    classifier = ShotClassifier()
    
    # Загрузка датасета AVE
    dataset_path = "datasets/AVE"
    (x_train, y_train), (x_val, y_val) = classifier.load_ave_dataset(dataset_path)
    
    # Обучение модели
    classifier.train((x_train, y_train), (x_val, y_val), epochs=30)
    
    # Сохранение модели
    classifier.save_model("backend/ai_service/trained_models/shot_classifier_model.h5") 