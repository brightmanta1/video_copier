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

# Если TensorFlow доступен, импортируем его через Docker
if TENSORFLOW_AVAILABLE:
    from backend.python.app.utils.tensorflow_docker import import_tensorflow
    tf = import_tensorflow()
    if tf is not None:
        logger.info(f"TensorFlow успешно импортирован через Docker: {tf.__version__}")
    else:
        logger.error("Не удалось импортировать TensorFlow через Docker")
        TENSORFLOW_AVAILABLE = False

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
        self.shot_types = [
            "wide_shot", "medium_shot", "close_up",
            "tracking_shot", "pan", "tilt", "zoom", "static"
        ]
        self.editing_patterns = {
            "establishing": ["wide_shot", "fade_in"],
            "action_sequence": ["cut", "medium_shot", "close_up"],
            "dramatic_reveal": ["zoom", "fade_in", "close_up"],
            "smooth_transition": ["dissolve", "tracking_shot"],
            "emphasis": ["flash", "close_up"],
            "scene_change": ["fade_out", "wide_shot", "fade_in"]
        }
        self.num_classes = len(self.effect_categories)
        self.num_shot_types = len(self.shot_types)
        self.initialized = False
        
        # Метрики качества монтажа
        self.quality_metrics = {
            "pacing": 0.0,  # Темп монтажа
            "shot_variety": 0.0,  # Разнообразие планов
            "transition_smoothness": 0.0,  # Плавность переходов
            "dramatic_impact": 0.0  # Драматический эффект
        }
        
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
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet', 
                include_top=False, 
                input_shape=self.input_shape[1:]
            )
            
            # Замораживаем часть слоев базовой модели
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            # Создаем модель для извлечения признаков из отдельных кадров
            frame_model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten()
            ])
            
            # Создаем последовательную модель для анализа изменений во времени
            sequence_model = tf.keras.Sequential([
                tf.keras.layers.TimeDistributed(frame_model, input_shape=self.input_shape),
                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.LSTM(128),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Компилируем модель
            sequence_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
            
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(self.model_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
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

    def analyze_editing_pattern(self, video_path):
        """
        Анализирует паттерны монтажа в видео.
        
        Args:
            video_path: Путь к видео
            
        Returns:
            Dict с анализом монтажных решений
        """
        try:
            patterns = []
            current_pattern = []
            
            # Получаем последовательность эффектов и планов
            effects, shots = self.detect_effects_and_shots(video_path)
            
            for i, (effect, shot) in enumerate(zip(effects, shots)):
                current_pattern.append((effect, shot))
                
                # Анализируем паттерн
                for pattern_name, expected_sequence in self.editing_patterns.items():
                    if self._match_pattern(current_pattern, expected_sequence):
                        patterns.append({
                            'name': pattern_name,
                            'timestamp': i * (1/30),  # Примерное время
                            'confidence': self._calculate_pattern_confidence(current_pattern)
                        })
                        current_pattern = []
                        break
            
            # Обновляем метрики качества
            self._update_quality_metrics(patterns, effects, shots)
            
            return {
                'patterns': patterns,
                'quality_metrics': self.quality_metrics,
                'suggestions': self._generate_editing_suggestions(patterns)
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе паттернов монтажа: {str(e)}")
            return None

    def _match_pattern(self, current_pattern, expected_sequence):
        """
        Сопоставляет текущую последовательность с ожидаемым паттерном.
        
        Args:
            current_pattern: Текущая последовательность эффектов и планов
            expected_sequence: Ожидаемый паттерн
            
        Returns:
            bool: Соответствует ли последовательность паттерну
        """
        if len(current_pattern) != len(expected_sequence):
            return False
            
        for (effect, shot), expected in zip(current_pattern, expected_sequence):
            if expected in self.effect_categories and effect != expected:
                return False
            if expected in self.shot_types and shot != expected:
                return False
                
        return True

    def _calculate_pattern_confidence(self, pattern):
        """
        Вычисляет уверенность в обнаруженном паттерне.
        
        Args:
            pattern: Последовательность эффектов и планов
            
        Returns:
            float: Уверенность в паттерне [0,1]
        """
        confidences = []
        for effect, shot in pattern:
            effect_conf = self.predict_confidence(effect)
            shot_conf = self.predict_shot_confidence(shot)
            confidences.extend([effect_conf, shot_conf])
            
        return np.mean(confidences)

    def _update_quality_metrics(self, patterns, effects, shots):
        """
        Обновляет метрики качества монтажа.
        
        Args:
            patterns: Обнаруженные паттерны
            effects: Последовательность эффектов
            shots: Последовательность планов
        """
        # Оценка темпа монтажа
        self.quality_metrics["pacing"] = self._calculate_pacing(effects)
        
        # Оценка разнообразия планов
        self.quality_metrics["shot_variety"] = len(set(shots)) / self.num_shot_types
        
        # Оценка плавности переходов
        self.quality_metrics["transition_smoothness"] = self._calculate_smoothness(effects)
        
        # Оценка драматического эффекта
        self.quality_metrics["dramatic_impact"] = self._calculate_dramatic_impact(patterns)

    def _generate_editing_suggestions(self, patterns):
        """
        Генерирует рекомендации по улучшению монтажа.
        
        Args:
            patterns: Обнаруженные паттерны
            
        Returns:
            List[str]: Список рекомендаций
        """
        suggestions = []
        
        # Анализ темпа
        if self.quality_metrics["pacing"] < 0.5:
            suggestions.append("Рекомендуется увеличить динамику монтажа, добавив больше склеек")
        elif self.quality_metrics["pacing"] > 0.8:
            suggestions.append("Рекомендуется снизить темп монтажа для лучшего восприятия")
            
        # Анализ разнообразия
        if self.quality_metrics["shot_variety"] < 0.4:
            suggestions.append("Используйте более разнообразные планы для поддержания интереса")
            
        # Анализ переходов
        if self.quality_metrics["transition_smoothness"] < 0.6:
            suggestions.append("Добавьте плавные переходы между резкими склейками")
            
        # Анализ драматургии
        if self.quality_metrics["dramatic_impact"] < 0.5:
            suggestions.append("Усильте драматический эффект использованием крупных планов и акцентных переходов")
            
        return suggestions

    def _calculate_pacing(self, effects):
        """
        Вычисляет оценку темпа монтажа.
        
        Args:
            effects: Последовательность эффектов
            
        Returns:
            float: Оценка темпа [0,1]
        """
        cut_frequency = effects.count("cut") / len(effects)
        effect_variety = len(set(effects)) / len(self.effect_categories)
        return (cut_frequency + effect_variety) / 2

    def _calculate_smoothness(self, effects):
        """
        Вычисляет оценку плавности переходов.
        
        Args:
            effects: Последовательность эффектов
            
        Returns:
            float: Оценка плавности [0,1]
        """
        smooth_transitions = ["dissolve", "fade_in", "fade_out"]
        smooth_count = sum(1 for effect in effects if effect in smooth_transitions)
        return smooth_count / len(effects)

    def _calculate_dramatic_impact(self, patterns):
        """
        Вычисляет оценку драматического эффекта.
        
        Args:
            patterns: Обнаруженные паттерны
            
        Returns:
            float: Оценка драматического эффекта [0,1]
        """
        dramatic_patterns = ["dramatic_reveal", "emphasis"]
        dramatic_count = sum(1 for p in patterns if p["name"] in dramatic_patterns)
        return min(1.0, dramatic_count / (len(patterns) + 1))

    def detect_effects_and_shots(self, video_path):
        """
        Определяет эффекты и типы планов в видео.
        
        Args:
            video_path: Путь к видео
            
        Returns:
            Tuple[List[str], List[str]]: Списки эффектов и типов планов
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            effects = []
            shots = []
            
            # Анализируем каждый кадр
            prev_frame = None
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Определяем тип плана
                shot_type = self._detect_shot_type(frame)
                shots.append(shot_type)
                
                # Определяем эффект между кадрами
                if prev_frame is not None:
                    effect = self._detect_effect_between_frames(prev_frame, frame)
                    effects.append(effect)
                    
                prev_frame = frame.copy()
            
            cap.release()
            return effects, shots
        except Exception as e:
            logger.error(f"Ошибка при анализе видео: {str(e)}")
            return [], []

    def _detect_shot_type(self, frame):
        """
        Определяет тип плана на основе анализа кадра.
        
        Args:
            frame: Кадр видео
            
        Returns:
            str: Тип плана
        """
        try:
            # Анализ размера объектов в кадре
            face_sizes = self._detect_faces(frame)
            if face_sizes:
                avg_face_size = np.mean(face_sizes)
                frame_size = frame.shape[0] * frame.shape[1]
                face_ratio = avg_face_size / frame_size
                
                # Определяем тип плана по размеру лиц
                if face_ratio > 0.15:
                    return "close_up"
                elif face_ratio > 0.05:
                    return "medium_shot"
                else:
                    return "wide_shot"
            
            # Если лица не найдены, анализируем общую композицию
            edges = cv2.Canny(frame, 100, 200)
            edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])
            
            if edge_density < 0.1:
                return "wide_shot"
            elif edge_density < 0.2:
                return "medium_shot"
            else:
                return "close_up"
                
        except Exception as e:
            logger.error(f"Ошибка при определении типа плана: {str(e)}")
            return "medium_shot"  # По умолчанию

    def _detect_faces(self, frame):
        """
        Определяет размеры лиц в кадре.
        
        Args:
            frame: Кадр видео
            
        Returns:
            List[float]: Список размеров обнаруженных лиц
        """
        try:
            # Используем каскады Хаара для обнаружения лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Конвертируем в градации серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Обнаруживаем лица
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Вычисляем размеры лиц
            face_sizes = [w * h for (x, y, w, h) in faces]
            
            return face_sizes
        except Exception as e:
            logger.error(f"Ошибка при обнаружении лиц: {str(e)}")
            return []

    def _detect_effect_between_frames(self, prev_frame, curr_frame):
        """
        Определяет эффект между двумя кадрами.
        
        Args:
            prev_frame: Предыдущий кадр
            curr_frame: Текущий кадр
            
        Returns:
            str: Тип эффекта
        """
        try:
            # Вычисляем разницу между кадрами
            diff = cv2.absdiff(prev_frame, curr_frame)
            diff_mean = np.mean(diff)
            
            # Определяем тип эффекта на основе характеристик разницы
            if diff_mean < 5:
                return "static"
            elif diff_mean > 100:
                return "cut"
            else:
                # Анализируем паттерн изменений
                hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
                hsv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
                
                # Проверяем изменение яркости
                brightness_diff = np.mean(hsv_curr[:,:,2]) - np.mean(hsv_prev[:,:,2])
                
                if abs(brightness_diff) > 50:
                    if brightness_diff > 0:
                        return "fade_in"
                    else:
                        return "fade_out"
                
                # Проверяем размытие
                laplacian_prev = cv2.Laplacian(prev_frame, cv2.CV_64F).var()
                laplacian_curr = cv2.Laplacian(curr_frame, cv2.CV_64F).var()
                
                if laplacian_curr < laplacian_prev * 0.5:
                    return "blur_transition"
                
                # Проверяем движение
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                
                if np.mean(mag) > 5:
                    # Анализируем направление движения
                    mean_angle = np.mean(ang)
                    if abs(mean_angle - np.pi) < 0.5:
                        return "push"
                    elif abs(mean_angle - np.pi/2) < 0.5:
                        return "slide"
                    else:
                        return "cross_zoom"
                
                return "dissolve"
                
        except Exception as e:
            logger.error(f"Ошибка при определении эффекта: {str(e)}")
            return "cut"  # По умолчанию

    def predict_confidence(self, effect_type):
        """
        Возвращает уверенность в предсказании эффекта.
        
        Args:
            effect_type: Тип эффекта
            
        Returns:
            float: Уверенность [0,1]
        """
        try:
            # В реальной системе здесь будет использоваться модель
            # Сейчас возвращаем фиксированные значения для демонстрации
            confidence_map = {
                "cut": 0.95,
                "dissolve": 0.85,
                "fade_in": 0.9,
                "fade_out": 0.9,
                "static": 0.98,
                "push": 0.8,
                "slide": 0.8,
                "cross_zoom": 0.75,
                "blur_transition": 0.7
            }
            return confidence_map.get(effect_type, 0.5)
        except Exception as e:
            logger.error(f"Ошибка при вычислении уверенности эффекта: {str(e)}")
            return 0.5

    def predict_shot_confidence(self, shot_type):
        """
        Возвращает уверенность в определении типа плана.
        
        Args:
            shot_type: Тип плана
            
        Returns:
            float: Уверенность [0,1]
        """
        try:
            # В реальной системе здесь будет использоваться модель
            # Сейчас возвращаем фиксированные значения для демонстрации
            confidence_map = {
                "wide_shot": 0.9,
                "medium_shot": 0.85,
                "close_up": 0.95,
                "tracking_shot": 0.8,
                "pan": 0.75,
                "tilt": 0.75,
                "zoom": 0.8,
                "static": 0.98
            }
            return confidence_map.get(shot_type, 0.5)
        except Exception as e:
            logger.error(f"Ошибка при вычислении уверенности типа плана: {str(e)}")
            return 0.5

    def load_combined_datasets(self, edit3k_path, ave_path):
        """
        Загружает и комбинирует данные из датасетов Edit3K и AVE.
        
        Args:
            edit3k_path: Путь к датасету Edit3K
            ave_path: Путь к датасету AVE
            
        Returns:
            Dict с объединенными данными
        """
        try:
            # Загружаем Edit3K
            edit3k_data = self.load_edit3k_dataset(edit3k_path)
            
            # Загружаем AVE
            with open(ave_path / "metadata.json", 'r') as f:
                ave_data = json.load(f)
            
            # Объединяем данные
            combined_data = {
                'transitions': edit3k_data,  # Данные о переходах
                'shot_composition': ave_data,  # Данные о композиции
                'editing_patterns': self._extract_editing_patterns(edit3k_data, ave_data)
            }
            
            logger.info("Датасеты успешно объединены")
            return combined_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасетов: {str(e)}")
            return None

    def _extract_editing_patterns(self, edit3k_data, ave_data):
        """
        Извлекает паттерны монтажа из комбинации датасетов.
        
        Args:
            edit3k_data: Данные Edit3K
            ave_data: Данные AVE
            
        Returns:
            Dict с паттернами монтажа
        """
        patterns = {
            'establishing_shots': [],  # Начальные планы сцен
            'action_sequences': [],    # Последовательности в экшн-сценах
            'dramatic_moments': [],    # Драматические моменты
            'transitions': [],         # Переходы между сценами
            'shot_progressions': []    # Прогрессии планов
        }
        
        try:
            # Анализируем последовательности из Edit3K
            for video in edit3k_data:
                effects = video['effects']
                
                # Ищем характерные последовательности эффектов
                for i in range(len(effects) - 2):
                    seq = effects[i:i+3]
                    
                    # Определяем тип последовательности
                    if self._is_establishing_sequence(seq):
                        patterns['establishing_shots'].append(seq)
                    elif self._is_action_sequence(seq):
                        patterns['action_sequences'].append(seq)
                    elif self._is_dramatic_sequence(seq):
                        patterns['dramatic_moments'].append(seq)
                        
            # Анализируем композиции из AVE
            for annotation in ave_data['annotations']:
                shot_seq = annotation['shot_sequence']
                
                # Анализируем прогрессии планов
                if len(shot_seq) >= 3:
                    patterns['shot_progressions'].append(shot_seq)
                
                # Анализируем переходы между сценами
                if 'transition_type' in annotation:
                    patterns['transitions'].append({
                        'from_shot': shot_seq[-1],
                        'to_shot': shot_seq[0],
                        'transition': annotation['transition_type']
                    })
            
            return patterns
        except Exception as e:
            logger.error(f"Ошибка при извлечении паттернов: {str(e)}")
            return patterns

    def _is_establishing_sequence(self, sequence):
        """Определяет, является ли последовательность устанавливающей"""
        establishing_patterns = [
            ['fade_in', 'wide_shot', 'static'],
            ['dissolve', 'wide_shot', 'pan'],
            ['fade_in', 'tracking_shot', 'wide_shot']
        ]
        return any(self._sequence_matches_pattern(sequence, pattern) 
                  for pattern in establishing_patterns)

    def _is_action_sequence(self, sequence):
        """Определяет, является ли последовательность экшн-сценой"""
        action_patterns = [
            ['cut', 'medium_shot', 'cut'],
            ['medium_shot', 'close_up', 'cut'],
            ['tracking_shot', 'cut', 'close_up']
        ]
        return any(self._sequence_matches_pattern(sequence, pattern) 
                  for pattern in action_patterns)

    def _is_dramatic_sequence(self, sequence):
        """Определяет, является ли последовательность драматической"""
        dramatic_patterns = [
            ['close_up', 'fade_out', 'fade_in'],
            ['zoom', 'close_up', 'static'],
            ['tracking_shot', 'close_up', 'fade_out']
        ]
        return any(self._sequence_matches_pattern(sequence, pattern) 
                  for pattern in dramatic_patterns)

    def _sequence_matches_pattern(self, sequence, pattern):
        """Проверяет соответствие последовательности паттерну"""
        return all(s == p or p in s or s in p 
                  for s, p in zip(sequence, pattern))

    def analyze_video_composition(self, video_path):
        """
        Анализирует композицию видео и предлагает улучшения.
        
        Args:
            video_path: Путь к видео
            
        Returns:
            Dict с анализом и рекомендациями
        """
        try:
            # Получаем последовательности эффектов и планов
            effects, shots = self.detect_effects_and_shots(video_path)
            
            # Анализируем паттерны монтажа
            patterns = self.analyze_editing_pattern(video_path)
            
            # Анализируем композицию
            composition_analysis = {
                'shot_distribution': self._analyze_shot_distribution(shots),
                'pacing_analysis': self._analyze_pacing(effects),
                'transition_analysis': self._analyze_transitions(effects),
                'dramatic_structure': self._analyze_dramatic_structure(patterns)
            }
            
            # Генерируем рекомендации
            recommendations = self._generate_composition_recommendations(
                composition_analysis, patterns
            )
            
            return {
                'analysis': composition_analysis,
                'patterns': patterns,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе композиции: {str(e)}")
            return None

    def _analyze_shot_distribution(self, shots):
        """Анализирует распределение типов планов"""
        try:
            distribution = {}
            total_shots = len(shots)
            
            # Подсчитываем частоту каждого типа плана
            for shot_type in self.shot_types:
                count = shots.count(shot_type)
                distribution[shot_type] = {
                    'count': count,
                    'percentage': count / total_shots if total_shots > 0 else 0
                }
            
            # Оцениваем разнообразие планов
            variety_score = len(set(shots)) / len(self.shot_types)
            
            return {
                'distribution': distribution,
                'variety_score': variety_score,
                'total_shots': total_shots
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе распределения планов: {str(e)}")
            return None

    def _analyze_pacing(self, effects):
        """Анализирует темп монтажа"""
        try:
            # Подсчитываем среднюю длительность между склейками
            cut_positions = [i for i, effect in enumerate(effects) 
                           if effect == 'cut']
            if len(cut_positions) > 1:
                cut_intervals = np.diff(cut_positions)
                avg_interval = np.mean(cut_intervals)
                std_interval = np.std(cut_intervals)
            else:
                avg_interval = len(effects)
                std_interval = 0
            
            # Оцениваем динамику
            dynamics = {
                'fast_cuts': sum(1 for interval in cut_intervals 
                               if interval < avg_interval/2),
                'slow_cuts': sum(1 for interval in cut_intervals 
                               if interval > avg_interval*2)
            }
            
            return {
                'average_interval': avg_interval,
                'interval_std': std_interval,
                'dynamics': dynamics
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе темпа: {str(e)}")
            return None

    def _analyze_transitions(self, effects):
        """Анализирует использование переходов"""
        try:
            transition_types = [effect for effect in effects 
                              if effect != 'cut' and effect != 'static']
            
            # Подсчитываем частоту каждого типа перехода
            transition_counts = {}
            for transition in transition_types:
                if transition not in transition_counts:
                    transition_counts[transition] = 0
                transition_counts[transition] += 1
            
            # Оцениваем разнообразие переходов
            variety_score = len(transition_counts) / len(self.effect_categories)
            
            return {
                'counts': transition_counts,
                'variety_score': variety_score,
                'total_transitions': len(transition_types)
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе переходов: {str(e)}")
            return None

    def _analyze_dramatic_structure(self, patterns):
        """Анализирует драматическую структуру"""
        try:
            # Анализируем паттерны для определения драматической структуры
            structure = {
                'establishing_shots': len(patterns.get('establishing_shots', [])),
                'dramatic_moments': len(patterns.get('dramatic_moments', [])),
                'action_sequences': len(patterns.get('action_sequences', [])),
                'pattern_distribution': {}
            }
            
            # Анализируем распределение паттернов
            total_patterns = sum(len(p) for p in patterns.values())
            for pattern_type, pattern_list in patterns.items():
                structure['pattern_distribution'][pattern_type] = {
                    'count': len(pattern_list),
                    'percentage': len(pattern_list) / total_patterns 
                                if total_patterns > 0 else 0
                }
            
            return structure
        except Exception as e:
            logger.error(f"Ошибка при анализе драматической структуры: {str(e)}")
            return None

    def _generate_composition_recommendations(self, analysis, patterns):
        """Генерирует рекомендации по улучшению композиции"""
        try:
            recommendations = []
            
            # Анализ распределения планов
            shot_dist = analysis['shot_distribution']
            if shot_dist['variety_score'] < 0.5:
                recommendations.append(
                    "Увеличьте разнообразие планов для поддержания интереса"
                )
            
            # Анализ темпа
            pacing = analysis['pacing_analysis']
            if pacing['interval_std'] > pacing['average_interval']:
                recommendations.append(
                    "Сделайте темп монтажа более равномерным для лучшего восприятия"
                )
            
            # Анализ переходов
            transitions = analysis['transition_analysis']
            if transitions['variety_score'] < 0.3:
                recommendations.append(
                    "Используйте более разнообразные переходы между сценами"
                )
            
            # Анализ драматической структуры
            dramatic = analysis['dramatic_structure']
            if dramatic['establishing_shots'] < 2:
                recommendations.append(
                    "Добавьте больше устанавливающих планов для лучшей ориентации зрителя"
                )
            
            return recommendations
        except Exception as e:
            logger.error(f"Ошибка при генерации рекомендаций: {str(e)}")
            return []


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