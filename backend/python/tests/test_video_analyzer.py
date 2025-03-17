import unittest
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))
from app.models.video_analyzer import VideoAnalyzer
from app.models.dataset_manager import DatasetManager

class TestVideoAnalyzer(unittest.TestCase):
    """Тесты для VideoAnalyzer"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем временный файл для тестового видео
        self.temp_video = self._create_test_video()
        
        # Инициализируем анализатор
        self.dataset_manager = DatasetManager()
        self.analyzer = VideoAnalyzer(self.dataset_manager)
        
    def tearDown(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_video):
            os.remove(self.temp_video)
    
    def _create_test_video(self):
        """Создание тестового видео для анализа"""
        try:
            from moviepy.editor import ColorClip
            
            # Создаем цветной клип с изменениями для тестирования определения границ сцен
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.close()
            
            # Создаем синий клип
            blue_clip = ColorClip(size=(640, 480), color=(0, 0, 255), duration=1)
            
            # Создаем красный клип
            red_clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=1)
            
            # Объединяем и сохраняем
            clips = [blue_clip, red_clip, blue_clip]
            from moviepy.editor import concatenate_videoclips
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(temp_file.name, fps=24)
            
            return temp_file.name
        except Exception as e:
            print(f"Ошибка создания тестового видео: {e}")
            # Если не удалось создать видео, вернем путь к пустому файлу
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.close()
            return temp_file.name
    
    def test_extract_frames(self):
        """Тест извлечения кадров"""
        # Проверяем существование файла перед тестом
        self.assertTrue(os.path.exists(self.temp_video), "Тестовое видео не существует")
        
        frames = self.analyzer.extract_frames(self.temp_video, sample_rate=10)
        self.assertIsInstance(frames, np.ndarray, "Результат должен быть numpy массивом")
        self.assertGreater(len(frames), 0, "Должен быть хотя бы один кадр")
    
    def test_detect_shot_boundaries(self):
        """Тест определения границ сцен"""
        # Генерируем тестовые кадры с явным изменением
        frames = np.array([
            np.ones((10, 10, 3)) * 50,  # Первый кадр
            np.ones((10, 10, 3)) * 50,  # Похожий на первый
            np.ones((10, 10, 3)) * 200, # Сильно отличающийся (граница сцены)
            np.ones((10, 10, 3)) * 200  # Похожий на предыдущий
        ])
        
        # Установим меньшее пороговое значение для теста
        self.analyzer.shot_threshold = 20
        
        boundaries = self.analyzer.detect_shot_boundaries(frames)
        
        # Должны быть обнаружены: начало, после второго кадра и конец
        self.assertEqual(len(boundaries), 3, "Должно быть 3 границы (начало, середина, конец)")
        self.assertEqual(boundaries[0], 0, "Первая граница должна быть в начале")
        self.assertEqual(boundaries[1], 2, "Вторая граница должна быть после второго кадра")
        self.assertEqual(boundaries[2], 4, "Последняя граница должна указывать на конец")
    
    def test_analyze_video(self):
        """Тест полного анализа видео"""
        result = self.analyzer.analyze_video(self.temp_video, use_gpu=False)
        
        # Проверяем структуру результата
        self.assertIn('total_frames', result, "Результат должен содержать total_frames")
        self.assertIn('shots', result, "Результат должен содержать shots")
        self.assertIn('shot_boundaries', result, "Результат должен содержать shot_boundaries")
        
        # Проверяем содержимое каждого shot
        if result['shots']:
            shot = result['shots'][0]
            self.assertIn('start_frame', shot, "Shot должен содержать start_frame")
            self.assertIn('end_frame', shot, "Shot должен содержать end_frame")
            self.assertIn('duration_frames', shot, "Shot должен содержать duration_frames")
            self.assertIn('features', shot, "Shot должен содержать features")

if __name__ == '__main__':
    unittest.main() 