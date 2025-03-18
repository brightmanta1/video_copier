"""
Модуль для работы с TensorFlow через Docker
Обеспечивает выполнение моделей TensorFlow в контейнере Docker
"""

import logging
import subprocess
import os
import json
import tempfile
from pathlib import Path
import time
import uuid

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('docker_tf_helper')

class TensorFlowInterface:
    """Эмуляция интерфейса TensorFlow для работы через Docker"""
    def __init__(self, docker_tf):
        self.docker_tf = docker_tf
        self.keras = self.KerasInterface(docker_tf)
        self.__version__ = self._get_version()
        
    def _get_version(self):
        """Получает версию TensorFlow из контейнера"""
        script = """
import tensorflow as tf
import json
print(json.dumps({"version": tf.__version__}))
        """
        result = self.docker_tf.execute_script(script)
        return result.get("version", "unknown") if result else "unknown"
        
    class KerasInterface:
        """Эмуляция интерфейса Keras"""
        def __init__(self, docker_tf):
            self.docker_tf = docker_tf
            self.models = self.ModelsInterface(docker_tf)
            self.layers = self.LayersInterface(docker_tf)
            self.optimizers = self.OptimizersInterface(docker_tf)
            self.callbacks = self.CallbacksInterface(docker_tf)
            self.applications = self.ApplicationsInterface(docker_tf)
            self.utils = self.UtilsInterface(docker_tf)
            
        class ModelsInterface:
            """Эмуляция интерфейса Keras Models"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def load_model(self, model_path, custom_objects=None):
                """Загружает модель через Docker"""
                return self.docker_tf.predict(model_path, None)
                
            def Sequential(self, layers=None):
                """Создает последовательную модель"""
                script = """
import tensorflow as tf
model = tf.keras.Sequential()
                """
                return self.docker_tf.execute_script(script)
                
        class LayersInterface:
            """Эмуляция интерфейса Keras Layers"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def Dense(self, units, activation=None, **kwargs):
                """Создает полносвязный слой"""
                return {"layer_type": "Dense", "units": units, "activation": activation, **kwargs}
                
            def Dropout(self, rate, **kwargs):
                """Создает слой Dropout"""
                return {"layer_type": "Dropout", "rate": rate, **kwargs}
                
            def GlobalAveragePooling2D(self, **kwargs):
                """Создает слой GlobalAveragePooling2D"""
                return {"layer_type": "GlobalAveragePooling2D", **kwargs}
                
        class OptimizersInterface:
            """Эмуляция интерфейса Keras Optimizers"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def Adam(self, learning_rate=0.001, **kwargs):
                """Создает оптимизатор Adam"""
                return {"optimizer_type": "Adam", "learning_rate": learning_rate, **kwargs}
                
        class CallbacksInterface:
            """Эмуляция интерфейса Keras Callbacks"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def ModelCheckpoint(self, filepath, **kwargs):
                """Создает callback ModelCheckpoint"""
                return {"callback_type": "ModelCheckpoint", "filepath": filepath, **kwargs}
                
            def EarlyStopping(self, **kwargs):
                """Создает callback EarlyStopping"""
                return {"callback_type": "EarlyStopping", **kwargs}
                
            def ReduceLROnPlateau(self, **kwargs):
                """Создает callback ReduceLROnPlateau"""
                return {"callback_type": "ReduceLROnPlateau", **kwargs}
                
        class ApplicationsInterface:
            """Эмуляция интерфейса Keras Applications"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def ResNet50(self, **kwargs):
                """Создает модель ResNet50"""
                return {"model_type": "ResNet50", **kwargs}
                
        class UtilsInterface:
            """Эмуляция интерфейса Keras Utils"""
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def to_categorical(self, y, num_classes=None):
                """Преобразует метки в one-hot вектора"""
                script = f"""
import tensorflow as tf
import numpy as np
import json

y = np.array({y.tolist() if hasattr(y, 'tolist') else y})
result = tf.keras.utils.to_categorical(y, num_classes={num_classes})
print(json.dumps({{"result": result.tolist()}}))
                """
                result = self.docker_tf.execute_script(script)
                return result.get("result") if result else None

class DockerTensorFlow:
    """Класс для работы с TensorFlow через Docker"""
    
    def __init__(self, image_name="tensorflow/tensorflow:2.12.0", mount_path=None):
        """
        Инициализация класса
        
        Args:
            image_name (str): имя образа Docker с TensorFlow
            mount_path (str): путь для монтирования внутрь контейнера
        """
        self.image_name = image_name
        self.container_id = None
        self.mount_path = mount_path or Path.cwd()
        self.temp_dir = None
        self.initialized = False
        self.tensorflow = None
        
    def __enter__(self):
        """Метод для использования с контекстным менеджером"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Метод для использования с контекстным менеджером"""
        self.cleanup()
        
    def initialize(self):
        """Инициализирует Docker контейнер с TensorFlow"""
        if self.initialized:
            return True
            
        # Проверяем Docker
        if not self._check_docker():
            logger.error("Docker недоступен")
            return False
            
        # Проверяем наличие образа
        if not self._check_image():
            logger.error(f"Образ {self.image_name} не найден и не может быть загружен")
            return False
            
        # Создаем временную директорию для файлов
        self.temp_dir = Path(tempfile.mkdtemp(prefix="docker_tf_"))
        logger.info(f"Создана временная директория: {self.temp_dir}")
        
        # Запускаем контейнер
        if not self._start_container():
            logger.error("Не удалось запустить контейнер")
            return False
            
        self.initialized = True
        return True
        
    def cleanup(self):
        """Освобождает ресурсы"""
        if self.container_id:
            try:
                logger.info(f"Останавливаем контейнер {self.container_id}")
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True,
                    check=False
                )
                self.container_id = None
            except Exception as e:
                logger.error(f"Ошибка при остановке контейнера: {str(e)}")
                
        if self.temp_dir and self.temp_dir.exists():
            try:
                # Удаляем временные файлы
                for file in self.temp_dir.glob("*"):
                    file.unlink()
                self.temp_dir.rmdir()
                logger.info(f"Удалена временная директория: {self.temp_dir}")
                self.temp_dir = None
            except Exception as e:
                logger.error(f"Ошибка при удалении временной директории: {str(e)}")
                
        self.initialized = False
        
    def _check_docker(self):
        """Проверяет доступность Docker"""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Docker установлен: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Docker не установлен или не запущен: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при проверке Docker: {str(e)}")
            return False
            
    def _check_image(self):
        """Проверяет наличие образа TensorFlow в Docker"""
        try:
            # Проверяем, есть ли образ локально
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", self.image_name], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if self.image_name in result.stdout:
                logger.info(f"Образ Docker с TensorFlow найден: {self.image_name}")
                return True
                
            # Если образа нет, пытаемся его загрузить
            logger.info(f"Загрузка образа {self.image_name}...")
            pull_result = subprocess.run(
                ["docker", "pull", self.image_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if pull_result.returncode == 0:
                logger.info(f"Образ {self.image_name} успешно загружен")
                return True
            else:
                # При ошибке загрузки GPU версии, пробуем версию CPU
                if "gpu" in self.image_name.lower():
                    cpu_image = self.image_name.replace("-gpu", "")
                    logger.info(f"Пробуем загрузить версию CPU: {cpu_image}")
                    
                    pull_result = subprocess.run(
                        ["docker", "pull", cpu_image],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if pull_result.returncode == 0:
                        logger.info(f"Образ {cpu_image} успешно загружен")
                        self.image_name = cpu_image
                        return True
                        
                logger.error(f"Не удалось загрузить образ: {pull_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при проверке образа Docker: {str(e)}")
            return False
            
    def _start_container(self):
        """Запускает контейнер Docker с TensorFlow"""
        try:
            # Проверяем, есть ли уже запущенный контейнер с нужным образом
            result = subprocess.run(
                ["docker", "ps", "--filter", f"ancestor={self.image_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout.strip():
                container_id = result.stdout.strip().split("\n")[0]
                logger.info(f"Найден запущенный контейнер с TensorFlow: {container_id}")
                self.container_id = container_id
                return True
                
            # Запускаем новый контейнер
            logger.info("Запуск нового контейнера с TensorFlow...")
            
            # Создаем уникальное имя контейнера
            container_name = f"tensorflow_{uuid.uuid4().hex[:8]}"
            
            # Запускаем контейнер в фоновом режиме
            # Монтируем текущую директорию и временную директорию
            result = subprocess.run(
                ["docker", "run", "-d", "--rm",
                 "--name", container_name,
                 "-v", f"{self.mount_path}:/app",
                 "-v", f"{self.temp_dir}:/tmp/tf_scripts",
                 "-p", "8501:8501",
                 self.image_name,
                 "python", "-c", "import time; import tensorflow as tf; "
                           "print(f'TensorFlow {tf.__version__} готов к использованию'); "
                           "time.sleep(3600*24)"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.container_id = result.stdout.strip()
                logger.info(f"Контейнер с TensorFlow успешно запущен: {self.container_id}")
                
                # Небольшая пауза, чтобы контейнер успел запуститься
                time.sleep(2)
                return True
            else:
                logger.error(f"Ошибка при запуске контейнера: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при запуске контейнера Docker: {str(e)}")
            return False
            
    def execute_script(self, script, script_args=None):
        """
        Выполняет Python скрипт в контейнере Docker
        
        Args:
            script (str): содержимое Python скрипта
            script_args (list): аргументы для скрипта
            
        Returns:
            dict or None: результат выполнения скрипта или None при ошибке
        """
        if not self.initialized and not self.initialize():
            return None
            
        if not self.container_id:
            logger.error("Контейнер не запущен")
            return None
            
        try:
            # Создаем временный файл со скриптом
            script_path = self.temp_dir / f"script_{uuid.uuid4().hex[:8]}.py"
            
            with open(script_path, 'w') as f:
                f.write(script)
                
            # Формируем команду для запуска скрипта
            cmd = ["docker", "exec", self.container_id, "python", f"/tmp/tf_scripts/{script_path.name}"]
            
            if script_args:
                cmd.extend(script_args)
                
            # Запускаем скрипт
            logger.info(f"Выполнение скрипта в контейнере...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Удаляем временный файл
            script_path.unlink()
            
            if result.returncode != 0:
                logger.error(f"Ошибка при выполнении скрипта: {result.stderr}")
                return None
                
            # Пытаемся распарсить вывод как JSON
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                # Если не получилось, возвращаем текстовый вывод
                return {"output": result.stdout.strip()}
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении скрипта в Docker: {str(e)}")
            return None
            
    def test_tensorflow(self):
        """
        Запускает тестовый скрипт для проверки TensorFlow
        
        Returns:
            bool: True если тест прошел успешно, иначе False
        """
        test_script = """
import tensorflow as tf
import numpy as np
import json

# Создаем простую модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy')

# Создаем тестовые данные
x = np.random.random((10, 5))
y = np.random.randint(0, 2, (10, 1))

# Проверяем предсказание
predictions = model.predict(x)

# Выводим результаты в формате JSON
result = {
    "tensorflow_version": tf.__version__,
    "model_summary": str(model.summary()),
    "predictions_shape": str(predictions.shape),
    "predictions_sample": predictions.tolist()[:3],
    "status": "ok"
}

print(json.dumps(result))
        """
        
        result = self.execute_script(test_script)
        
        if result and result.get("status") == "ok":
            logger.info(f"TensorFlow версия: {result.get('tensorflow_version')}")
            logger.info(f"Форма предсказаний: {result.get('predictions_shape')}")
            logger.info("TensorFlow в Docker успешно протестирован!")
            return True
        
        logger.error("Тест TensorFlow не пройден")
        return False
        
    def train_model(self, model_script, model_path=None, data=None):
        """
        Тренирует модель в контейнере Docker
        
        Args:
            model_script (str): скрипт для обучения модели
            model_path (str): путь для сохранения модели
            data (dict): данные для обучения модели
            
        Returns:
            dict or None: результат обучения или None при ошибке
        """
        if not self.initialized and not self.initialize():
            return None
            
        # Путь для сохранения модели в контейнере
        if model_path:
            abs_model_path = Path(model_path).resolve()
            container_model_path = f"/app/{abs_model_path.relative_to(self.mount_path)}"
        else:
            # Создаем временный файл для модели
            model_filename = f"model_{uuid.uuid4().hex[:8]}.h5"
            abs_model_path = self.temp_dir / model_filename
            container_model_path = f"/tmp/tf_scripts/{model_filename}"
            
        # Если есть данные, сохраняем их во временный файл
        data_path = None
        if data:
            data_filename = f"data_{uuid.uuid4().hex[:8]}.json"
            data_path = self.temp_dir / data_filename
            
            with open(data_path, 'w') as f:
                json.dump(data, f)
                
            container_data_path = f"/tmp/tf_scripts/{data_filename}"
        else:
            container_data_path = None
            
        # Добавляем пути к скрипту
        final_script = f"""
import os
# Устанавливаем пути для модели и данных
MODEL_PATH = "{container_model_path}"
DATA_PATH = {f'"{container_data_path}"' if container_data_path else None}

# Создаем директорию для модели, если её нет
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

{model_script}
        """
        
        # Выполняем скрипт в контейнере
        result = self.execute_script(final_script)
        
        # Очищаем временные файлы с данными
        if data_path and data_path.exists():
            data_path.unlink()
            
        # Возвращаем результат
        return {
            "model_path": str(abs_model_path),
            "result": result
        }
        
    def predict(self, model_path, input_data, predict_script=None):
        """
        Выполняет предсказание на основе модели
        
        Args:
            model_path (str): путь к файлу модели
            input_data (dict): входные данные для предсказания
            predict_script (str): скрипт для предсказания (опционально)
            
        Returns:
            dict or None: результат предсказания или None при ошибке
        """
        if not self.initialized and not self.initialize():
            return None
            
        # Преобразуем пути в абсолютные
        abs_model_path = Path(model_path).resolve()
        
        # Проверяем, существует ли файл модели
        if not abs_model_path.exists():
            logger.error(f"Файл модели не найден: {abs_model_path}")
            return None
            
        # Путь к модели внутри контейнера
        try:
            container_model_path = f"/app/{abs_model_path.relative_to(self.mount_path)}"
        except ValueError:
            logger.error(f"Путь к модели должен быть внутри {self.mount_path}")
            return None
            
        # Сохраняем входные данные во временный файл
        data_filename = f"predict_data_{uuid.uuid4().hex[:8]}.json"
        data_path = self.temp_dir / data_filename
        
        with open(data_path, 'w') as f:
            json.dump(input_data, f)
            
        container_data_path = f"/tmp/tf_scripts/{data_filename}"
        
        # Если скрипт для предсказания не предоставлен, используем стандартный
        if not predict_script:
            predict_script = """
import tensorflow as tf
import numpy as np
import json

# Загружаем модель
model = tf.keras.models.load_model(MODEL_PATH)

# Загружаем данные
with open(DATA_PATH, 'r') as f:
    input_data = json.load(f)
    
# Преобразуем данные в numpy массив
if isinstance(input_data, list):
    x = np.array(input_data)
else:
    x = np.array([input_data])
    
# Выполняем предсказание
predictions = model.predict(x)

# Возвращаем результат
result = {
    "predictions": predictions.tolist(),
    "status": "ok"
}

print(json.dumps(result))
            """
        
        # Добавляем пути к скрипту
        final_script = f"""
# Устанавливаем пути для модели и данных
MODEL_PATH = "{container_model_path}"
DATA_PATH = "{container_data_path}"

{predict_script}
        """
        
        # Выполняем скрипт в контейнере
        result = self.execute_script(final_script)
        
        # Удаляем временный файл с данными
        data_path.unlink()
        
        return result

    def get_tensorflow(self):
        """Возвращает интерфейс TensorFlow"""
        if not self.tensorflow:
            self.tensorflow = TensorFlowInterface(self)
        return self.tensorflow

def get_tensorflow_docker():
    """
    Получает экземпляр DockerTensorFlow
    
    Returns:
        DockerTensorFlow: экземпляр класса DockerTensorFlow
    """
    return DockerTensorFlow()

def run_tensorflow_script(script, args=None):
    """
    Запускает Python скрипт в контейнере Docker с TensorFlow
    
    Args:
        script (str): содержимое Python скрипта
        args (list): аргументы для скрипта
        
    Returns:
        dict or None: результат выполнения скрипта или None при ошибке
    """
    with DockerTensorFlow() as docker_tf:
        return docker_tf.execute_script(script, args)

def test_tensorflow_docker():
    """
    Проверяет работу TensorFlow в Docker
    
    Returns:
        bool: True если тест прошел успешно, иначе False
    """
    with DockerTensorFlow() as docker_tf:
        return docker_tf.test_tensorflow()

def train_model_in_docker(model_script, model_path=None, data=None):
    """
    Тренирует модель в контейнере Docker
    
    Args:
        model_script (str): скрипт для обучения модели
        model_path (str): путь для сохранения модели
        data (dict): данные для обучения модели
        
    Returns:
        dict or None: результат обучения или None при ошибке
    """
    with DockerTensorFlow() as docker_tf:
        return docker_tf.train_model(model_script, model_path, data)

def predict_with_model_in_docker(model_path, input_data, predict_script=None):
    """
    Выполняет предсказание на основе модели в контейнере Docker
    
    Args:
        model_path (str): путь к файлу модели
        input_data (dict): входные данные для предсказания
        predict_script (str): скрипт для предсказания (опционально)
        
    Returns:
        dict or None: результат предсказания или None при ошибке
    """
    with DockerTensorFlow() as docker_tf:
        return docker_tf.predict(model_path, input_data, predict_script)

if __name__ == "__main__":
    # Пример использования
    success = test_tensorflow_docker()
    print(f"Тест TensorFlow в Docker: {'успешно' if success else 'не удалось'}") 