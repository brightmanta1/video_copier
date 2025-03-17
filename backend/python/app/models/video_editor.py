from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx import all as vfx
from typing import Dict, List, Optional, Tuple, Any
import os
import uuid
import cv2
import logging
import tempfile
from pathlib import Path

# Простой импорт ffmpeg, пакет ffmpeg удален, ffmpeg-python установлен
import ffmpeg

# Импортируем утилиты из нашего приложения
from ..utils.video_utils import VideoUtils

logger = logging.getLogger('VideoEditor')

class VideoEditor:
    """Класс для применения структуры редактирования к видео"""
    
    def __init__(self, output_dir='output/', use_ffmpeg_direct=True):
        """
        Инициализация редактора видео
        
        Args:
            output_dir: директория для сохранения результатов
            use_ffmpeg_direct: использовать прямые вызовы ffmpeg для оптимизации операций
        """
        self.output_dir = output_dir
        self.use_ffmpeg_direct = use_ffmpeg_direct
            
        self.temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def apply_structure(self, target_video_path: str, edit_structure: Dict, 
                       optimize_large_files: bool = True) -> str:
        """
        Применение структуры редактирования к целевому видео
        
        Args:
            target_video_path: путь к целевому видео
            edit_structure: структура редактирования
            optimize_large_files: оптимизировать работу с большими файлами
            
        Returns:
            путь к отредактированному видео
        """
        # Получаем информацию о видео
        try:
            video_info = VideoUtils.get_video_info(target_video_path)
            duration = video_info['duration']
            filesize_mb = video_info['filesize_mb']
        except Exception as e:
            logger.warning(f"Ошибка при получении информации о видео: {e}. Используем стандартные параметры.")
            duration = 0
            filesize_mb = 0
            
        # Определяем стратегию обработки в зависимости от размера файла
        use_segmented_approach = optimize_large_files and (filesize_mb > 500 or duration > 300)
        
        # Определяем путь для выходного файла
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Для больших файлов используем сегментированный подход
        if use_segmented_approach and self.use_ffmpeg_direct:
            return self._apply_structure_segmented(target_video_path, edit_structure, output_path)
        else:
            return self._apply_structure_moviepy(target_video_path, edit_structure, output_path)
    
    def _apply_structure_moviepy(self, target_video_path: str, edit_structure: Dict, output_path: str) -> str:
        """
        Применение структуры редактирования с использованием MoviePy
        
        Args:
            target_video_path: путь к целевому видео
            edit_structure: структура редактирования
            output_path: путь для сохранения результата
            
        Returns:
            путь к отредактированному видео
        """
        # Загрузка целевого видео
        target_clip = VideoFileClip(target_video_path)
        
        # Подготовка списка сегментов для финального видео
        edited_segments = []
        
        # Обработка структуры
        for shot in edit_structure['shots']:
            # Рассчитываем временные границы для целевого видео
            start_ratio = shot['start_frame'] / edit_structure['total_frames']
            end_ratio = shot['end_frame'] / edit_structure['total_frames']
            
            target_start = start_ratio * target_clip.duration
            target_end = end_ratio * target_clip.duration
            
            # Создаем сегмент
            segment = target_clip.subclip(target_start, target_end)
            
            # Применяем эффекты если они есть
            if 'effects' in shot:
                segment = self._apply_effects(segment, shot['effects'])
            
            edited_segments.append(segment)
        
        # Склеиваем все сегменты
        if not edited_segments:
            return target_video_path  # Возвращаем оригинал если нет изменений
            
        final_clip = concatenate_videoclips(edited_segments)
        
        # Сохраняем результат
        final_clip.write_videofile(output_path, codec='libx264')
        
        # Закрываем все клипы
        target_clip.close()
        for segment in edited_segments:
            segment.close()
        final_clip.close()
        
        return output_path
        
    def _apply_structure_segmented(self, target_video_path: str, edit_structure: Dict, output_path: str) -> str:
        """
        Применение структуры редактирования с использованием ffmpeg для больших файлов
        
        Args:
            target_video_path: путь к целевому видео
            edit_structure: структура редактирования
            output_path: путь для сохранения результата
            
        Returns:
            путь к отредактированному видео
        """
        # Получаем информацию о видео
        video_info = VideoUtils.get_video_info(target_video_path)
        
        # Создаем временную директорию для сегментов
        temp_dir = os.path.join(self.temp_dir, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        
        # Список файлов сегментов для финальной склейки
        segment_files = []
        segment_list_file = os.path.join(temp_dir, "segments.txt")
        
        try:
            # Обрабатываем каждый кадр в структуре
            for i, shot in enumerate(edit_structure['shots']):
                # Рассчитываем временные границы для целевого видео
                start_ratio = shot['start_frame'] / edit_structure['total_frames']
                end_ratio = shot['end_frame'] / edit_structure['total_frames']
                
                target_start = start_ratio * video_info['duration']
                target_end = end_ratio * video_info['duration']
                
                # Создаем имя для сегмента
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                
                # Вырезаем сегмент с использованием ffmpeg
                VideoUtils.cut_video_segment(
                    target_video_path,
                    segment_path,
                    target_start,
                    target_end,
                    use_ffmpeg_direct=True
                )
                
                # Если есть эффекты - применяем их
                if 'effects' in shot:
                    # Для эффектов нам нужен MoviePy, поэтому загружаем клип
                    segment_clip = VideoFileClip(segment_path)
                    segment_clip = self._apply_effects(segment_clip, shot['effects'])
                    
                    # Сохраняем обработанный сегмент
                    processed_path = os.path.join(temp_dir, f"processed_{i:04d}.mp4")
                    segment_clip.write_videofile(processed_path, codec='libx264')
                    segment_clip.close()
                    
                    # Обновляем путь к сегменту
                    segment_path = processed_path
                    
                segment_files.append(segment_path)
                
            # Создаем файл со списком сегментов для ffmpeg
            with open(segment_list_file, 'w') as f:
                for segment in segment_files:
                    f.write(f"file '{os.path.abspath(segment)}'\n")
                    
            # Склеиваем все сегменты с использованием ffmpeg
            (
                ffmpeg
                .input(segment_list_file, format='concat', safe=0)
                .output(output_path, c='copy')
                .run(quiet=True, overwrite_output=True)
            )
                
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при сегментированной обработке видео: {e}")
            # Если произошла ошибка, пытаемся использовать стандартный подход
            logger.info("Пробуем применить стандартный подход с MoviePy")
            return self._apply_structure_moviepy(target_video_path, edit_structure, output_path)
            
        finally:
            # Удаляем временные файлы
            for segment in segment_files:
                if os.path.exists(segment):
                    try:
                        os.remove(segment)
                    except:
                        pass
                        
            if os.path.exists(segment_list_file):
                try:
                    os.remove(segment_list_file)
                except:
                    pass
    
    def _apply_effects(self, clip, effects: List[Dict]) -> VideoFileClip:
        """
        Применение эффектов к клипу
        
        Args:
            clip: видеоклип
            effects: список эффектов с параметрами
            
        Returns:
            обработанный видеоклип
        """
        for effect in effects:
            effect_type = effect.get('type')
            params = effect.get('params', {})
            
            # Применяем соответствующий эффект в зависимости от типа
            if effect_type == 'brightness':
                factor = params.get('factor', 1.5)
                clip = clip.fx(vfx.colorx, factor)
                
            elif effect_type == 'contrast':
                factor = params.get('factor', 1.3)
                clip = clip.fx(vfx.lum_contrast, contrast=factor)
                
            elif effect_type == 'saturation':
                # Эффект насыщенности требует специальной обработки
                factor = params.get('factor', 1.5)
                def saturation(image):
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = hsv[:, :, 1] * factor
                    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                clip = clip.fl_image(saturation)
                
            elif effect_type == 'grayscale':
                clip = clip.fx(vfx.blackwhite)
                
            elif effect_type == 'sepia':
                clip = clip.fx(vfx.sepia)
                
            elif effect_type == 'blur':
                radius = params.get('radius', 2)
                clip = clip.fx(vfx.blur, radius)
                
            elif effect_type == 'fade_in':
                duration = params.get('duration', 0.5)
                clip = clip.fx(vfx.fadein, duration)
                
            elif effect_type == 'fade_out':
                duration = params.get('duration', 0.5)
                clip = clip.fx(vfx.fadeout, duration)
                
            elif effect_type == 'speed':
                factor = params.get('factor', 1.5)
                clip = clip.fx(vfx.speedx, factor)
                
            # Добавьте здесь другие эффекты по необходимости
            
        return clip
        
    def apply_lut(self, video_path: str, lut_path: str, strength: float = 1.0) -> str:
        """
        Применение LUT (Look-Up Table) к видео для цветокоррекции
        
        Args:
            video_path: путь к исходному видео
            lut_path: путь к файлу LUT (.cube)
            strength: сила применения эффекта (0.0-1.0)
            
        Returns:
            путь к обработанному видео
        """
        # Проверяем наличие файлов
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT файл не найден: {lut_path}")
            
        # Определяем путь для выходного файла
        output_filename = f"{uuid.uuid4()}_lut.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        if self.use_ffmpeg_direct:
            try:
                # Применяем LUT с помощью ffmpeg
                (
                    ffmpeg
                    .input(video_path)
                    .filter('lut3d', file=lut_path, interp='tetrahedral')
                    .output(output_path, **{'c:a': 'copy'})
                    .run(quiet=True, overwrite_output=True)
                )
                return output_path
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при применении LUT через ffmpeg: {e}. Переключаемся на MoviePy.")
        
        # Используем MoviePy как запасной вариант
        clip = VideoFileClip(video_path)
        
        # Применяем LUT через OpenCV для каждого кадра
        def apply_lut_frame(image):
            # Здесь должен быть код для применения LUT через OpenCV
            # Это упрощенная версия, в реальности нужно парсить файл .cube и применять его
            # как 3D LUT к каждому кадру
            return image  # Заглушка, в реальности здесь будет преобразование
            
        processed_clip = clip.fl_image(apply_lut_frame)
        processed_clip.write_videofile(output_path, codec='libx264')
        
        clip.close()
        processed_clip.close()
        
        return output_path 