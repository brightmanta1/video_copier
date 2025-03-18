"""
Простой тест для проверки Docker TensorFlow без зависимостей от приложения
"""

import logging
import sys
import os
import json
import subprocess
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_tensorflow_simple')

def check_docker():
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

def check_tensorflow_docker_image():
    """Проверяет наличие образа TensorFlow в Docker"""
    image_name = "tensorflow/tensorflow:2.12.0-gpu"
    
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", image_name], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if image_name in result.stdout:
            logger.info(f"Образ Docker с TensorFlow найден: {image_name}")
            return True, image_name
        else:
            logger.warning(f"Образ Docker с TensorFlow не найден: {image_name}")
            logger.info("Попытка загрузки образа TensorFlow из Docker Hub...")
            
            # Пытаемся загрузить образ с TensorFlow
            pull_result = subprocess.run(
                ["docker", "pull", image_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if pull_result.returncode == 0:
                logger.info(f"Образ {image_name} успешно загружен")
                return True, image_name
            else:
                logger.error(f"Не удалось загрузить образ {image_name}: {pull_result.stderr}")
                
                # Пробуем версию без GPU
                cpu_image = "tensorflow/tensorflow:2.12.0"
                logger.info(f"Пробуем загрузить версию без GPU: {cpu_image}")
                
                pull_result = subprocess.run(
                    ["docker", "pull", cpu_image],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if pull_result.returncode == 0:
                    logger.info(f"Образ {cpu_image} успешно загружен")
                    return True, cpu_image
                else:
                    logger.error(f"Не удалось загрузить образ {cpu_image}: {pull_result.stderr}")
                    return False, None
    except Exception as e:
        logger.error(f"Ошибка при проверке образа Docker: {str(e)}")
        return False, None

def ensure_tensorflow_container_running(image_name):
    """Убеждается, что контейнер с TensorFlow запущен"""
    try:
        # Проверяем, есть ли уже запущенный контейнер с TensorFlow
        result = subprocess.run(
            ["docker", "ps", "--filter", f"ancestor={image_name}", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout.strip():
            container_id = result.stdout.strip().split("\n")[0]
            logger.info(f"Найден запущенный контейнер с TensorFlow: {container_id}")
            return True, container_id
            
        # Запускаем новый контейнер
        logger.info("Запуск нового контейнера с TensorFlow...")
        
        # Определяем директорию проекта для монтирования
        project_root = Path.cwd()
        
        # Запускаем контейнер в фоновом режиме
        result = subprocess.run(
            ["docker", "run", "-d", "--rm",
             "-v", f"{project_root}:/app",
             "-p", "8501:8501",
             image_name,
             "python", "-c", "import time; import tensorflow as tf; print(f'TensorFlow {tf.__version__} готов к использованию'); time.sleep(3600*24)"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            container_id = result.stdout.strip()
            logger.info(f"Контейнер с TensorFlow успешно запущен: {container_id}")
            return True, container_id
        else:
            logger.error(f"Ошибка при запуске контейнера: {result.stderr}")
            return False, None
            
    except Exception as e:
        logger.error(f"Ошибка при обеспечении работы контейнера Docker: {str(e)}")
        return False, None

def run_tensorflow_test(container_id):
    """Запускает тест TensorFlow в контейнере"""
    try:
        # Создаем временный Python скрипт для тестирования TensorFlow
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
    "predictions_shape": predictions.shape,
    "predictions_sample": predictions.tolist()[:3],
    "status": "ok"
}

print(json.dumps(result))
        """
        
        # Определяем директорию проекта
        project_root = Path.cwd()
        test_script_path = project_root / "temp_tensorflow_test.py"
        
        # Записываем скрипт во временный файл
        with open(test_script_path, 'w') as f:
            f.write(test_script)
        
        # Запускаем скрипт в контейнере Docker
        result = subprocess.run(
            ["docker", "exec", container_id, 
             "python", "/app/temp_tensorflow_test.py"],
            capture_output=True, 
            text=True, 
            check=False
        )
        
        # Удаляем временный скрипт
        os.remove(test_script_path)
        
        if result.returncode != 0:
            logger.error(f"Ошибка при выполнении скрипта в Docker: {result.stderr}")
            return False
        
        # Анализируем результат
        try:
            test_result = json.loads(result.stdout.strip())
            
            logger.info(f"TensorFlow версия: {test_result['tensorflow_version']}")
            logger.info(f"Форма предсказаний: {test_result['predictions_shape']}")
            logger.info(f"Примеры предсказаний: {test_result['predictions_sample']}")
            
            logger.info("TensorFlow в Docker успешно протестирован!")
            return True
        except json.JSONDecodeError:
            logger.error(f"Не удалось декодировать результат: {result.stdout}")
            return False
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании TensorFlow: {str(e)}")
        return False

def main():
    """Основная функция теста"""
    logger.info("Начало теста Docker TensorFlow")
    
    # Шаг 1: Проверяем доступность Docker
    if not check_docker():
        logger.error("Docker недоступен. Установите и запустите Docker Desktop.")
        return False
    
    # Шаг 2: Проверяем/загружаем образ TensorFlow
    success, image_name = check_tensorflow_docker_image()
    if not success:
        logger.error("Не удалось найти или загрузить образ TensorFlow.")
        return False
    
    # Шаг 3: Убеждаемся, что контейнер с TensorFlow запущен
    success, container_id = ensure_tensorflow_container_running(image_name)
    if not success:
        logger.error("Не удалось запустить контейнер с TensorFlow.")
        return False
    
    # Шаг 4: Запускаем тест TensorFlow
    success = run_tensorflow_test(container_id)
    
    logger.info("Тест Docker TensorFlow завершен.")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 