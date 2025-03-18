import os
import subprocess
import logging
import importlib.util
import platform
import sys
from pathlib import Path

logger = logging.getLogger('TensorFlowDocker')

class TensorFlowDocker:
    """
    Класс для управления интеграцией TensorFlow через Docker
    """
    
    _instance = None
    _tensorflow_available = None
    _docker_available = None
    _docker_tensorflow_available = None
    _initialization_checked = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TensorFlowDocker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialization_checked:
            self._check_tensorflow()
            self._check_docker()
            self._initialization_checked = True
    
    def _check_tensorflow(self):
        """Проверка наличия TensorFlow в системе"""
        try:
            # Проверяем, установлен ли TensorFlow локально
            if importlib.util.find_spec("tensorflow") is not None:
                import tensorflow as tf
                logger.info(f"TensorFlow установлен локально, версия: {tf.__version__}")
                self._tensorflow_available = True
            else:
                logger.warning("TensorFlow не установлен локально")
                self._tensorflow_available = False
        except Exception as e:
            logger.error(f"Ошибка при проверке TensorFlow: {str(e)}")
            self._tensorflow_available = False
    
    def _check_docker(self):
        """Проверка наличия Docker и образа TensorFlow"""
        try:
            # Проверяем, установлен ли Docker
            result = subprocess.run(["docker", "--version"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
            
            if result.returncode == 0:
                logger.info(f"Docker установлен: {result.stdout.strip()}")
                self._docker_available = True
                
                # Проверяем, существует ли образ TensorFlow
                result = subprocess.run(["docker", "images", "--format", "{{.Repository}}", "tensorflow_app"], 
                                        capture_output=True, 
                                        text=True, 
                                        check=False)
                
                if "tensorflow_app" in result.stdout:
                    logger.info("Образ Docker с TensorFlow найден")
                    self._docker_tensorflow_available = True
                else:
                    logger.warning("Образ Docker с TensorFlow не найден")
                    self._docker_tensorflow_available = False
            else:
                logger.warning("Docker не установлен или не запущен")
                self._docker_available = False
                self._docker_tensorflow_available = False
        except Exception as e:
            logger.error(f"Ошибка при проверке Docker: {str(e)}")
            self._docker_available = False
            self._docker_tensorflow_available = False
    
    def is_tensorflow_available(self):
        """Проверяет, доступен ли TensorFlow (локально или через Docker)"""
        return self._tensorflow_available or self._docker_tensorflow_available
    
    def ensure_tensorflow_available(self):
        """Убеждается, что TensorFlow доступен, используя Docker если необходимо"""
        if self._tensorflow_available:
            return True
        
        if not self._docker_available:
            logger.error("TensorFlow недоступен локально, а Docker не установлен")
            return False
        
        if not self._docker_tensorflow_available:
            # Пытаемся собрать образ Docker с TensorFlow
            logger.info("Попытка сборки образа Docker с TensorFlow...")
            try:
                subprocess.run(["docker", "build", "-t", "tensorflow_app", "."], 
                               check=True, 
                               cwd=str(Path(__file__).resolve().parents[3]))  # Путь к корню проекта
                
                self._docker_tensorflow_available = True
                logger.info("Образ Docker с TensorFlow успешно собран")
                return True
            except Exception as e:
                logger.error(f"Ошибка при сборке образа Docker с TensorFlow: {str(e)}")
                return False
        
        return True
    
    def run_tensorflow_function(self, function_module, function_name, *args, **kwargs):
        """
        Запускает функцию TensorFlow локально или в Docker
        
        Args:
            function_module: модуль, содержащий функцию
            function_name: имя функции
            *args, **kwargs: аргументы функции
            
        Returns:
            результат выполнения функции
        """
        if self._tensorflow_available:
            # Если TensorFlow доступен локально, запускаем функцию напрямую
            try:
                module = importlib.import_module(function_module)
                func = getattr(module, function_name)
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка при выполнении функции TensorFlow локально: {str(e)}")
                return None
        
        elif self._docker_tensorflow_available:
            # Если TensorFlow доступен через Docker, запускаем функцию в контейнере
            try:
                # Формируем команду для запуска в Docker
                project_root = Path(__file__).resolve().parents[3]  # Путь к корню проекта
                
                # Сериализуем аргументы в формат, который можно передать в командной строке
                import json
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
module = importlib.import_module('{function_module}')
func = getattr(module, '{function_name}')

# Выполняем функцию и выводим результат
result = func(*args, **kwargs)
print(json.dumps(result))
                    """)
                
                # Запускаем скрипт в Docker
                result = subprocess.run(
                    ["docker", "run", "--rm", 
                     "-v", f"{project_root}:/app", 
                     "tensorflow_app", 
                     "python", "/app/temp_tensorflow_runner.py"],
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                
                # Удаляем временный скрипт
                os.remove(script_path)
                
                # Анализируем результат
                return json.loads(result.stdout.strip())
                
            except Exception as e:
                logger.error(f"Ошибка при выполнении функции TensorFlow в Docker: {str(e)}")
                return None
        else:
            logger.error("TensorFlow недоступен ни локально, ни через Docker")
            return None


# Создаем экземпляр-одиночку
tensorflow_docker = TensorFlowDocker()

# Функция-обертка для импорта TensorFlow
def import_tensorflow():
    """
    Импортирует TensorFlow, используя локальную установку или Docker
    
    Returns:
        модуль TensorFlow или None, если он недоступен
    """
    if tensorflow_docker.is_tensorflow_available():
        if TensorFlowDocker._tensorflow_available:
            # Если TensorFlow доступен локально, импортируем его напрямую
            import tensorflow as tf
            return tf
        elif tensorflow_docker.ensure_tensorflow_available():
            # Создаем прокси-объект для TensorFlow
            # Это заглушка, которая решает проблему с "import tensorflow could not be resolved"
            # При вызове методов будет использоваться Docker
            class TensorFlowProxy:
                def __getattr__(self, name):
                    def method(*args, **kwargs):
                        return tensorflow_docker.run_tensorflow_function("tensorflow", name, *args, **kwargs)
                    return method
            
            return TensorFlowProxy()
    
    # Возвращаем заглушку, чтобы избежать ошибок импорта
    logger.warning("Возвращается заглушка TensorFlow. Функциональность ограничена.")
    class TensorFlowDummy:
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                logger.error(f"TensorFlow недоступен: вызов {name}(*{args}, **{kwargs}) невозможен")
                return None
            return dummy_method
    
    return TensorFlowDummy() 