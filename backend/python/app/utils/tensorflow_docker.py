import os
import subprocess
import logging
import json
import time
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TensorFlowDocker')

class TensorFlowDocker:
    """
    Класс для автоматической подгрузки TensorFlow исключительно из Docker
    """
    
    _instance = None
    _docker_available = None
    _docker_tensorflow_available = None
    _initialization_checked = False
    _docker_container_id = None
    _docker_container_running = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TensorFlowDocker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialization_checked:
            logger.info("TensorFlow будет использоваться только из Docker")
            self._check_docker()
            self._initialization_checked = True
            
        # Эмулируем интерфейс TensorFlow
        self.keras = self.KerasInterface(self)
        self.__version__ = self._get_tf_version()
        
        # Добавляем атрибуты TensorFlow
        self.constant = lambda x: x  # Простая эмуляция tf.constant
        self.float32 = 'float32'     # Тип данных
        self.int32 = 'int32'         # Тип данных
        
        # Добавляем вспомогательные функции
        self.cast = lambda x, dtype: x  # Эмуляция tf.cast
        self.reshape = lambda x, shape: x  # Эмуляция tf.reshape
    
    def _get_tf_version(self):
        """Получает версию TensorFlow из контейнера"""
        if not self.ensure_tensorflow_available():
            return "unknown"
            
        try:
            result = subprocess.run(
                ["docker", "exec", self._docker_container_id, "python", "-c",
                 "import tensorflow as tf; print(tf.__version__)"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def _check_docker(self):
        """Проверка наличия Docker и контейнера TensorFlow"""
        try:
            # Проверяем, установлен ли Docker
            result = subprocess.run(["docker", "--version"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
            
            if result.returncode == 0:
                logger.info(f"Docker установлен: {result.stdout.strip()}")
                self._docker_available = True
                
                # Проверяем запущенные контейнеры TensorFlow
                result = subprocess.run(
                    ["docker", "ps"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if "tensorflow/tensorflow" in result.stdout:
                    # Получаем ID контейнера
                    for line in result.stdout.split('\n'):
                        if "tensorflow/tensorflow" in line:
                            container_id = line.split()[0]
                            logger.info(f"Найден запущенный контейнер с TensorFlow: {container_id}")
                            self._docker_container_id = container_id
                            self._docker_container_running = True
                            self._docker_tensorflow_available = True
                            
                            # Проверяем версию TensorFlow в контейнере
                            version_check = subprocess.run(
                                ["docker", "exec", container_id, "python", "-c", 
                                 "import tensorflow as tf; print(f'TensorFlow {tf.__version__} готов к использованию')"],
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            
                            if version_check.returncode == 0:
                                logger.info(version_check.stdout.strip())
                            else:
                                logger.error(f"Ошибка при проверке версии TensorFlow: {version_check.stderr}")
                            break
                else:
                    logger.error("Не найдены запущенные контейнеры TensorFlow")
                    self._docker_tensorflow_available = False
            else:
                logger.error("Docker не установлен или не запущен")
                self._docker_available = False
                self._docker_tensorflow_available = False
        except Exception as e:
            logger.error(f"Ошибка при проверке Docker: {str(e)}")
            self._docker_available = False
            self._docker_tensorflow_available = False
    
    def is_tensorflow_available(self):
        """Проверяет, доступен ли TensorFlow через Docker"""
        return self._docker_available and self._docker_tensorflow_available and self._docker_container_running
    
    def ensure_tensorflow_available(self):
        """Убеждается, что TensorFlow доступен через Docker"""
        if not self._docker_available:
            logger.error("Docker не установлен или не запущен")
            return False
        
        if not self._docker_tensorflow_available or not self._docker_container_running:
            logger.error("Контейнер с TensorFlow не найден или не запущен")
            return False
            
        return True
    
    def run_keras_function(self, function_path, *args, **kwargs):
        """Запускает функцию keras через Docker"""
        return self.run_tensorflow_function('keras.' + function_path, None, *args, **kwargs)
        
    def run_tensorflow_function(self, function_module, function_name, *args, **kwargs):
        """
        Запускает функцию TensorFlow в Docker
        
        Args:
            function_module: путь к функции (например, 'keras.models.load_model')
            function_name: имя функции (не используется, оставлен для обратной совместимости)
            *args, **kwargs: аргументы функции
            
        Returns:
            результат выполнения функции
        """
        if not self.ensure_tensorflow_available():
            logger.error("TensorFlow через Docker недоступен")
            return None
            
        try:
            # Формируем команду для запуска в Docker
            args_json = json.dumps([args, kwargs])
            
            script = f"""
import tensorflow as tf
import json
import numpy as np

args, kwargs = json.loads('{args_json}')
result = tf.{function_module}(*args, **kwargs)

if hasattr(result, 'numpy'):
    result = result.numpy().tolist()
elif isinstance(result, np.ndarray):
    result = result.tolist()

print(json.dumps({{'result': result}}))
            """
            
            result = subprocess.run(
                ["docker", "exec", self._docker_container_id, "python", "-c", script],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при выполнении функции в Docker: {result.stderr}")
                return None
            
            try:
                output = json.loads(result.stdout)
                return output.get('result')
            except json.JSONDecodeError:
                logger.error(f"Ошибка при разборе результата: {result.stdout}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении функции в Docker: {str(e)}")
            return None

    class KerasInterface:
        """Эмуляция интерфейса Keras"""
        def __init__(self, docker_tf):
            self.docker_tf = docker_tf
            self.models = self.ModelsInterface(docker_tf)
            self.layers = self.LayersInterface(docker_tf)
            self.applications = self.ApplicationsInterface(docker_tf)
            self.optimizers = self.OptimizersInterface(docker_tf)
            self.callbacks = self.CallbacksInterface(docker_tf)
            self.utils = self.UtilsInterface(docker_tf)
            
            # Добавляем атрибуты Keras
            self.Sequential = self.models.Sequential
            self.Model = self.models.Model
            
        class ModelsInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def load_model(self, model_path, custom_objects=None):
                return self.docker_tf.run_keras_function('models.load_model', model_path, custom_objects)
                
            def Model(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('models.Model', *args, **kwargs)
                
            def Sequential(self, layers=None):
                return self.docker_tf.run_keras_function('models.Sequential', layers)
                
        class LayersInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def Dense(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.Dense', *args, **kwargs)
                
            def Dropout(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.Dropout', *args, **kwargs)
                
            def Conv2D(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.Conv2D', *args, **kwargs)
                
            def MaxPooling2D(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.MaxPooling2D', *args, **kwargs)
                
            def BatchNormalization(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.BatchNormalization', *args, **kwargs)
                
            def Flatten(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.Flatten', *args, **kwargs)
                
            def TimeDistributed(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.TimeDistributed', *args, **kwargs)
                
            def LSTM(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('layers.LSTM', *args, **kwargs)
                
        class ApplicationsInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def MobileNetV2(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('applications.MobileNetV2', *args, **kwargs)
                
            def ResNet50(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('applications.ResNet50', *args, **kwargs)
                
        class OptimizersInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def Adam(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('optimizers.Adam', *args, **kwargs)
                
        class CallbacksInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def ModelCheckpoint(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('callbacks.ModelCheckpoint', *args, **kwargs)
                
            def EarlyStopping(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('callbacks.EarlyStopping', *args, **kwargs)
                
            def ReduceLROnPlateau(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('callbacks.ReduceLROnPlateau', *args, **kwargs)
                
        class UtilsInterface:
            def __init__(self, docker_tf):
                self.docker_tf = docker_tf
                
            def to_categorical(self, *args, **kwargs):
                return self.docker_tf.run_keras_function('utils.to_categorical', *args, **kwargs)

def import_tensorflow():
    """
    Импортирует TensorFlow через Docker
    
    Returns:
        TensorFlowDocker: объект, эмулирующий интерфейс TensorFlow
    """
    tf_docker = TensorFlowDocker()
    if tf_docker.is_tensorflow_available():
        return tf_docker
    return None 