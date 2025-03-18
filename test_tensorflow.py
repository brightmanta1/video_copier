import logging
import sys
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_tensorflow')

# Добавляем корневую директорию проекта в путь для импорта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Импортируем функцию для получения TensorFlow
from backend.python.app.utils.tensorflow_docker import import_tensorflow

def test_tensorflow_import():
    """Тестирует импорт TensorFlow через Docker"""
    logger.info("Пытаемся импортировать TensorFlow...")
    
    # Импортируем TensorFlow
    tf = import_tensorflow()
    
    if tf is None:
        logger.error("Не удалось импортировать TensorFlow")
        return False
    
    # Проверяем работоспособность TensorFlow
    try:
        # Создаем простую модель
        logger.info("Создаем тестовую модель...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Компилируем модель
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Создаем тестовые данные
        import numpy as np
        x = np.random.random((10, 5))
        y = np.random.randint(0, 2, (10, 1))
        
        # Проверяем предсказание
        logger.info("Выполняем предсказание...")
        predictions = model.predict(x)
        
        logger.info(f"Предсказания: {predictions[:3]}...")
        logger.info("TensorFlow через Docker работает корректно!")
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании TensorFlow: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_tensorflow_import()
    sys.exit(0 if success else 1) 