import os
import sys
import git
import json
import subprocess
import shutil
import argparse
import logging
from pathlib import Path

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_trainer')

class ModelTrainer:
    """
    Класс для клонирования репозиториев датасетов, подготовки данных и обучения моделей
    """
    def __init__(self, output_dir="backend/ai_service/trained_models"):
        self.output_dir = output_dir
        self.datasets_dir = "datasets"
        self.repos = {
            "AVE": "https://github.com/dawitmureja/AVE.git",
            "Edit3K": "https://github.com/GX77/Edit3K.git"
        }
        
        # Создание директорий
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
    def clone_repositories(self, force=False):
        """
        Клонирует GitHub репозитории с датасетами
        
        Args:
            force: Принудительно заново клонировать даже если директория существует
        """
        for dataset_name, repo_url in self.repos.items():
            dataset_path = os.path.join(self.datasets_dir, dataset_name)
            
            if os.path.exists(dataset_path) and not force:
                logger.info(f"Репозиторий {dataset_name} уже загружен в {dataset_path}")
                continue
                
            if os.path.exists(dataset_path) and force:
                logger.info(f"Удаление существующей директории {dataset_path}")
                shutil.rmtree(dataset_path)
                
            logger.info(f"Клонирование репозитория {repo_url} в {dataset_path}")
            git.Repo.clone_from(repo_url, dataset_path)
            logger.info(f"Репозиторий {dataset_name} успешно клонирован")
    
    def process_datasets(self):
        """
        Обрабатывает загруженные датасеты и подготавливает их для обучения
        """
        # Обработка AVE датасета
        ave_path = os.path.join(self.datasets_dir, "AVE")
        if os.path.exists(ave_path):
            logger.info("Обработка датасета AVE...")
            self._process_ave_dataset(ave_path)
            
        # Обработка Edit3K датасета
        edit3k_path = os.path.join(self.datasets_dir, "Edit3K")
        if os.path.exists(edit3k_path):
            logger.info("Обработка датасета Edit3K...")
            self._process_edit3k_dataset(edit3k_path)
    
    def _process_ave_dataset(self, dataset_path):
        """
        Обработка датасета AVE: извлечение категорий, аннотаций, создание метаданных
        
        Args:
            dataset_path: Путь к директории датасета AVE
        """
        # Проверяем наличие файлов и директорий в AVE
        categories_file = os.path.join(dataset_path, "categories.txt")
        annotations_dir = os.path.join(dataset_path, "annotations")
        videos_dir = os.path.join(dataset_path, "videos")
        
        # Если нет необходимых файлов, считаем что датасет уже обработан
        if not (os.path.exists(categories_file) or 
                os.path.exists(annotations_dir) or 
                os.path.exists(videos_dir)):
            logger.warning("Не найдены исходные файлы AVE для обработки. Пропускаем...")
            return
        
        # Создаем метаданные на основе аннотаций
        metadata = {
            "dataset_name": "AVE",
            "version": "1.0.0",
            "description": "Audio-Visual Event Dataset для анализа видеофрагментов",
            "categories": [],
            "samples": []
        }
        
        # Загружаем категории
        if os.path.exists(categories_file):
            with open(categories_file, 'r') as f:
                categories_data = f.read().strip().split('\n')
                metadata["categories"] = categories_data
        
        # Обрабатываем аннотации
        if os.path.exists(annotations_dir):
            annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
            for ann_file in annotation_files:
                video_id = ann_file.split('.')[0]
                
                with open(os.path.join(annotations_dir, ann_file), 'r') as f:
                    lines = f.read().strip().split('\n')
                    
                    sample = {
                        "id": video_id,
                        "filename": f"{video_id}.mp4",
                        "annotations": []
                    }
                    
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            category = parts[2].strip()
                            
                            annotation = {
                                "start": start_time,
                                "end": end_time,
                                "category": category,
                                "confidence": 0.95
                            }
                            
                            sample["annotations"].append(annotation)
                    
                    metadata["samples"].append(sample)
        
        # Сохраняем метаданные
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Метаданные AVE созданы в {metadata_path}")
    
    def _process_edit3k_dataset(self, dataset_path):
        """
        Обработка датасета Edit3K: извлечение эффектов, создание метаданных
        
        Args:
            dataset_path: Путь к директории датасета Edit3K
        """
        # Проверяем наличие файлов в Edit3K
        annotations_file = os.path.join(dataset_path, "annotations.json")
        videos_dir = os.path.join(dataset_path, "videos")
        
        # Если нет необходимых файлов, считаем что датасет уже обработан
        if not (os.path.exists(annotations_file) or os.path.exists(videos_dir)):
            logger.warning("Не найдены исходные файлы Edit3K для обработки. Пропускаем...")
            return
        
        # Создаем метаданные на основе аннотаций
        metadata = {
            "dataset_name": "Edit3K",
            "version": "2.0.0",
            "description": "Датасет для анализа эффектов видеоредактирования и переходов",
            "effect_categories": [
                "cut", "dissolve", "fade_in", "fade_out", "wipe", 
                "slide", "push", "flash", "cross_zoom", "rotation", 
                "blur_transition", "color_transfer"
            ],
            "samples": []
        }
        
        # Загружаем аннотации
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
                
                for video_id, video_data in annotations_data.items():
                    sample = {
                        "id": video_id,
                        "filename": f"{video_id}.mp4",
                        "effects": []
                    }
                    
                    if "effects" in video_data:
                        for effect in video_data["effects"]:
                            effect_data = {
                                "start": effect.get("start_time", 0),
                                "end": effect.get("end_time", 0),
                                "type": effect.get("type", "unknown"),
                                "parameters": effect.get("parameters", {})
                            }
                            
                            sample["effects"].append(effect_data)
                    
                    metadata["samples"].append(sample)
        
        # Подсчитываем статистику
        metadata["total_videos"] = len(metadata["samples"])
        metadata["total_effects"] = sum(len(s["effects"]) for s in metadata["samples"])
        
        # Сохраняем метаданные
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Метаданные Edit3K созданы в {metadata_path}")
    
    def train_models(self):
        """
        Обучение моделей на подготовленных датасетах
        """
        # Проверяем наличие датасетов
        ave_path = os.path.join(self.datasets_dir, "AVE")
        edit3k_path = os.path.join(self.datasets_dir, "Edit3K")
        
        # Путь к модулям
        models_dir = "backend/ai_service/models"
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Обучение модели для типов кадров (AVE)
        shot_classifier_path = os.path.join(models_dir, "shot_classifier.py")
        if os.path.exists(shot_classifier_path) and os.path.exists(ave_path):
            logger.info("Запуск обучения модели ShotClassifier...")
            try:
                subprocess.run([
                    sys.executable, 
                    shot_classifier_path
                ], check=True)
                logger.info("Обучение ShotClassifier завершено успешно")
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка при обучении ShotClassifier: {e}")
        
        # Обучение модели для эффектов видео (Edit3K)
        effect_detector_path = os.path.join(models_dir, "effect_detector.py")
        if os.path.exists(effect_detector_path) and os.path.exists(edit3k_path):
            logger.info("Запуск обучения модели EffectDetector...")
            try:
                subprocess.run([
                    sys.executable, 
                    effect_detector_path
                ], check=True)
                logger.info("Обучение EffectDetector завершено успешно")
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка при обучении EffectDetector: {e}")
    
    def run_pipeline(self, force_clone=False):
        """
        Запускает полный процесс подготовки и обучения
        
        Args:
            force_clone: Принудительно заново клонировать репозитории
        """
        logger.info("Запуск пайплайна обучения моделей...")
        
        # 1. Клонирование репозиториев
        self.clone_repositories(force=force_clone)
        
        # 2. Обработка датасетов
        self.process_datasets()
        
        # 3. Обучение моделей
        self.train_models()
        
        logger.info("Пайплайн обучения моделей завершен")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение моделей для VideoСopier")
    parser.add_argument("--force-clone", action="store_true", help="Принудительное клонирование репозиториев")
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    trainer.run_pipeline(force_clone=args.force_clone) 