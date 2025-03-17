#!/usr/bin/env python
"""
Скрипт для запуска процесса обучения моделей машинного обучения
для проекта Video Copier. Загружает данные из репозиториев GitHub,
обрабатывает их и обучает модели для детекции типов кадров и эффектов.
"""

import os
import sys
import argparse
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training_runner')

# Добавляем текущий каталог в путь поиска модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем класс ModelTrainer
try:
    from models.model_trainer import ModelTrainer
except ImportError as e:
    logger.error(f"Ошибка импорта: {e}")
    logger.error("Убедитесь, что вы запускаете скрипт из корневой директории проекта")
    sys.exit(1)

def main():
    """
    Основная функция для запуска процесса обучения
    """
    parser = argparse.ArgumentParser(description="Запуск обучения моделей для Video Copier")
    parser.add_argument("--force-clone", action="store_true", help="Принудительное клонирование репозиториев")
    parser.add_argument("--output-dir", default="backend/ai_service/trained_models", help="Директория для сохранения моделей")
    parser.add_argument("--skip-clone", action="store_true", help="Пропустить клонирование репозиториев")
    parser.add_argument("--skip-processing", action="store_true", help="Пропустить обработку датасетов")
    parser.add_argument("--train-shot-only", action="store_true", help="Обучить только модель типов кадров")
    parser.add_argument("--train-effect-only", action="store_true", help="Обучить только модель эффектов")
    args = parser.parse_args()
    
    try:
        # Создаем объект ModelTrainer
        trainer = ModelTrainer(output_dir=args.output_dir)
        
        # Клонирование репозиториев
        if not args.skip_clone:
            logger.info("Шаг 1: Клонирование репозиториев датасетов")
            trainer.clone_repositories(force=args.force_clone)
        else:
            logger.info("Шаг 1: Клонирование репозиториев пропущено")
            
        # Обработка датасетов
        if not args.skip_processing:
            logger.info("Шаг 2: Обработка датасетов")
            trainer.process_datasets()
        else:
            logger.info("Шаг 2: Обработка датасетов пропущена")
            
        # Обучение моделей
        logger.info("Шаг 3: Обучение моделей")
        
        if args.train_shot_only:
            logger.info("Обучение только модели типов кадров (ShotClassifier)")
            # Импортируем и запускаем обучение только ShotClassifier
            from models.shot_classifier import ShotClassifier
            classifier = ShotClassifier()
            dataset_path = "datasets/AVE"
            (x_train, y_train), (x_val, y_val) = classifier.load_ave_dataset(dataset_path)
            classifier.train((x_train, y_train), (x_val, y_val), epochs=30)
            classifier.save_model(os.path.join(args.output_dir, "shot_classifier_model.h5"))
        
        elif args.train_effect_only:
            logger.info("Обучение только модели эффектов (EffectDetector)")
            # Импортируем и запускаем обучение только EffectDetector
            from models.effect_detector import EffectDetector
            detector = EffectDetector()
            dataset_path = "datasets/Edit3K"
            (x_train, y_train), (x_val, y_val) = detector.load_edit3k_dataset(dataset_path)
            detector.train((x_train, y_train), (x_val, y_val), epochs=30)
            detector.save_model(os.path.join(args.output_dir, "effect_detector_model.h5"))
        
        else:
            # Обучаем обе модели
            trainer.train_models()
            
        logger.info("Обучение моделей успешно завершено!")
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 