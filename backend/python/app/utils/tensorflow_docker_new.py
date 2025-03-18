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
    
    def run_tensorflow_function(self, function_module, function_name, *args, **kwargs):
        """
        Запускает функцию TensorFlow в Docker
        
        Args:
            function_module: модуль, содержащий функцию
            function_name: имя функции
            *args, **kwargs: аргументы функции
            
        Returns:
            результат выполнения функции
        """
        # Проверяем доступность Docker TensorFlow
        if not self.ensure_tensorflow_available():
            logger.error("TensorFlow через Docker недоступен")
            return None
            
        # Запускаем функцию в контейнере Docker
        try:
            # Формируем команду для запуска в Docker
            project_root = Path(__file__).resolve().parents[3]  # Путь к корню проекта
            
            # Сериализуем аргументы в формат, который можно передать в командной строке
            args_json = json.dumps([args, kwargs])
            
            # Создаем временный Python скрипт для выполнения функции
            script_path = project_root / "temp_tensorflow_runner.py"
            with open(script_path, 'w') as f:
                f.write(f"""
import sys
import json
import importlib

# Загружаем аргументы
args_json = '{args_json}'
args, kwargs = json.loads(args_json)

# Импортируем модуль и получаем функцию
try:
    module = importlib.import_module('{function_module}')
    func = getattr(module, '{function_name}')

    # Выполняем функцию и выводим результат
    result = func(*args, **kwargs)
    print(json.dumps({{'status': 'success', 'result': result}}))
except Exception as e:
    print(json.dumps({{'status': 'error', 'error': str(e)}}))
                """)
            
            # Запускаем скрипт в контейнере Docker
            result = subprocess.run(
                ["docker", "exec", self._docker_container_id, 
                 "python", "/app/temp_tensorflow_runner.py"],
                capture_output=True, 
                text=True, 
                check=False
            )
            
            # Удаляем временный скрипт
            os.remove(script_path)
            
            if result.returncode != 0:
                logger.error(f"Ошибка при выполнении скрипта в Docker: {result.stderr}")
                return None
            
            # Анализируем результат
            try:
                output = json.loads(result.stdout)
                if output['status'] == 'success':
                    return output['result']
                else:
                    logger.error(f"Ошибка при выполнении функции: {output['error']}")
                    return None
            except json.JSONDecodeError:
                logger.error(f"Ошибка при разборе результата: {result.stdout}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении функции в Docker: {str(e)}")
            return None

def import_tensorflow():
    """
    Импортирует TensorFlow через Docker
    
    Returns:
        объект TensorFlow или None, если импорт не удался
    """
    tf_docker = TensorFlowDocker()
    if tf_docker.is_tensorflow_available():
        return tf_docker
    return None 