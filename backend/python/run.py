#!/usr/bin/env python
"""
Точка входа для запуска приложения Video Copier
"""
import os
import sys
import logging
from pathlib import Path

# Настройка путей
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "app.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('run')

def main():
    """Основная функция запуска приложения"""
    logger.info("Запуск Video Copier...")
    
    try:
        # Импортируем приложение
        from app import create_app, get_config
        
        # Получаем конфигурацию
        config = get_config()
        
        # Создаем и запускаем приложение
        app = create_app()
        
        logger.info(f"Запуск сервера на http://{config.HOST}:{config.PORT}")
        
        # Запуск Flask приложения
        app.run(
            host=config.HOST, 
            port=config.PORT,
            debug=config.DEBUG
        )
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей: {e}")
        logger.error("Убедитесь, что все зависимости установлены")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Создание необходимых директорий
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    
    # Запуск приложения
    main() 