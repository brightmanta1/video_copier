from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import os
import logging
import uuid
import shutil
import tempfile
from pathlib import Path

# Простой импорт ffmpeg, пакет ffmpeg удален, ffmpeg-python установлен
import ffmpeg

# Импортируем утилиты из нашего приложения
from ..utils.video_utils import VideoUtils

logger = logging.getLogger('EffectsLibrary')

class EffectsLibrary:
    """Библиотека эффектов для применения к видео"""
    
    def __init__(self, temp_dir=None, use_ffmpeg_direct=True):
        """
        Инициализация библиотеки эффектов
        
        Args:
            temp_dir: директория для временных файлов
            use_ffmpeg_direct: использовать прямые вызовы ffmpeg для оптимизации операций
        """
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), 'video_effects')
        self.use_ffmpeg_direct = use_ffmpeg_direct
            
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def apply_effect(self, clip: VideoFileClip, effect_type: str, params: Dict[str, Any] = None) -> VideoFileClip:
        """
        Применяет указанный эффект к видеоклипу через MoviePy
        
        Args:
            clip: VideoFileClip для обработки
            effect_type: тип эффекта
            params: параметры эффекта (опционально)
            
        Returns:
            Обработанный VideoFileClip
        """
        params = params or {}
        
        # Видео эффекты
        if effect_type == 'brightness':
            factor = params.get('factor', 1.5)
            return clip.fx(vfx.colorx, factor)
            
        elif effect_type == 'contrast':
            factor = params.get('factor', 1.3)
            return vfx.lum_contrast(clip, contrast=factor)
            
        elif effect_type == 'saturation':
            factor = params.get('factor', 1.5)
            # Ручная реализация насыщенности через HSV
            def saturation(frame):
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * factor
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return clip.fl_image(saturation)
        
        # Фильтры
        elif effect_type == 'grayscale':
            return clip.fx(vfx.blackwhite)
            
        elif effect_type == 'sepia':
            return clip.fx(vfx.sepia)
            
        elif effect_type == 'blur':
            radius = params.get('radius', 2)
            return clip.fx(vfx.blur, radius)
        
        # Анимации
        elif effect_type == 'fade_in':
            duration = params.get('duration', 0.5)
            return clip.fx(vfx.fadein, duration)
            
        elif effect_type == 'fade_out':
            duration = params.get('duration', 0.5)
            return clip.fx(vfx.fadeout, duration)
            
        elif effect_type == 'zoom_in':
            duration = params.get('duration', 1.0)
            zoom_factor = params.get('factor', 1.5)
            
            def zoom(t):
                if t < duration:
                    scale = 1 + (zoom_factor - 1) * t / duration
                    return scale
                return zoom_factor
                
            return clip.fx(vfx.resize, lambda t: zoom(t))
            
        elif effect_type == 'speed':
            factor = params.get('factor', 1.5)
            return clip.fx(vfx.speedx, factor)
            
        elif effect_type == 'reverse':
            return clip.fx(vfx.time_mirror)
            
        elif effect_type == 'rotate':
            angle = params.get('angle', 90)
            return clip.fx(vfx.rotate, angle)
            
        # Если эффект не найден, возвращаем оригинал
        return clip
        
    def apply_effect_to_file(self, video_path: str, effect_type: str, params: Dict[str, Any] = None) -> str:
        """
        Применяет эффект к видеофайлу с использованием ffmpeg для высокопроизводительных операций
        
        Args:
            video_path: путь к видео
            effect_type: тип эффекта
            params: параметры эффекта (опционально)
            
        Returns:
            путь к обработанному видео
        """
        params = params or {}
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
        # Создаем временный файл для результата
        output_filename = f"{uuid.uuid4()}_{effect_type}.mp4"
        output_path = os.path.join(self.temp_dir, output_filename)
        
        # Если можем использовать прямой вызов ffmpeg и эффект поддерживается
        if self.use_ffmpeg_direct:
            try:
                # Создаем базовый ffmpeg input
                stream = ffmpeg.input(video_path)
                
                # Видео эффекты через ffmpeg
                if effect_type == 'brightness':
                    factor = params.get('factor', 1.5)
                    # Используем фильтр eq для яркости
                    stream = stream.filter('eq', brightness=factor-1)
                
                elif effect_type == 'contrast':
                    factor = params.get('factor', 1.3)
                    # Используем фильтр eq для контраста
                    stream = stream.filter('eq', contrast=factor)
                
                elif effect_type == 'saturation':
                    factor = params.get('factor', 1.5)
                    # Используем фильтр eq для насыщенности
                    stream = stream.filter('eq', saturation=factor)
                
                elif effect_type == 'grayscale':
                    # Преобразование в оттенки серого
                    stream = stream.filter('hue', s=0)
                
                elif effect_type == 'sepia':
                    # Эффект сепии через colorchannelmixer
                    stream = stream.filter(
                        'colorchannelmixer', 
                        rr=0.393, rg=0.769, rb=0.189, 
                        gr=0.349, gg=0.686, gb=0.168, 
                        br=0.272, bg=0.534, bb=0.131
                    )
                
                elif effect_type == 'blur':
                    radius = params.get('radius', 2)
                    # Размытие по Гауссу
                    stream = stream.filter('boxblur', radius)
                
                elif effect_type == 'speed':
                    factor = params.get('factor', 1.5)
                    # Изменение скорости через setpts
                    stream = stream.filter('setpts', f'PTS/{factor}')
                    
                    # Если есть аудио, корректируем его скорость
                    audio = ffmpeg.input(video_path).audio
                    audio = audio.filter('atempo', min(2.0, factor))  # atempo поддерживает до 2x
                    
                    # Если нужно больше 2x, применяем несколько раз
                    if factor > 2.0:
                        remaining_factor = factor / 2.0
                        audio = audio.filter('atempo', min(2.0, remaining_factor))
                    
                    # Выходной файл с обработанным видео и аудио
                    stream = ffmpeg.output(stream, audio, output_path)
                
                elif effect_type == 'reverse':
                    # Реверс видео
                    stream = stream.filter('reverse')
                    
                    # Реверс аудио
                    audio = ffmpeg.input(video_path).audio.filter('areverse')
                    
                    # Выходной файл с обработанным видео и аудио
                    stream = ffmpeg.output(stream, audio, output_path)
                
                elif effect_type == 'rotate':
                    angle = params.get('angle', 90)
                    # Поворот видео
                    if angle == 90:
                        stream = stream.filter('transpose', 1)
                    elif angle == 180:
                        stream = stream.filter('transpose', 1).filter('transpose', 1)
                    elif angle == 270:
                        stream = stream.filter('transpose', 2)
                    else:
                        # Для произвольного угла используем rotate
                        stream = stream.filter('rotate', angle * np.pi / 180)
                
                elif effect_type == 'fade_in':
                    duration = params.get('duration', 0.5)
                    # Плавное появление
                    stream = stream.filter('fade', type='in', start_time=0, duration=duration)
                
                elif effect_type == 'fade_out':
                    duration = params.get('duration', 0.5)
                    
                    # Получаем длительность видео
                    probe = ffmpeg.probe(video_path)
                    video_duration = float(probe['format']['duration'])
                    
                    # Плавное исчезновение
                    stream = stream.filter('fade', type='out', start_time=video_duration-duration, duration=duration)
                
                # Если не было специальной обработки выше, создаем выходной поток
                if not output_path in str(stream):
                    stream = ffmpeg.output(stream, output_path)
                
                # Выполняем ffmpeg команду
                stream.run(quiet=True, overwrite_output=True)
                return output_path
                
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при применении эффекта {effect_type} через ffmpeg: {e}. Переключаемся на MoviePy.")
        
        # Резервный вариант через MoviePy
        try:
            clip = VideoFileClip(video_path)
            processed_clip = self.apply_effect(clip, effect_type, params)
            processed_clip.write_videofile(output_path, codec='libx264')
            clip.close()
            processed_clip.close()
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при применении эффекта {effect_type}: {e}")
            # Если произошла ошибка, возвращаем исходный файл
            return video_path
        
    def get_available_effects(self) -> Dict[str, List[str]]:
        """
        Возвращает список доступных эффектов
        
        Returns:
            словарь с категориями эффектов
        """
        return {
            'video_effects': ['brightness', 'contrast', 'saturation', 'speed', 'reverse', 'rotate'],
            'filters': ['grayscale', 'sepia', 'blur'],
            'animations': ['fade_in', 'fade_out', 'zoom_in'],
            'transitions': ['fade', 'wipe', 'slide']
        }
        
    def apply_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                        transition_type: str, duration: float = 1.0) -> VideoFileClip:
        """
        Применяет переход между двумя клипами через MoviePy
        
        Args:
            clip1: первый клип
            clip2: второй клип
            transition_type: тип перехода
            duration: длительность перехода
            
        Returns:
            Клип с применённым переходом
        """
        from moviepy.editor import concatenate_videoclips
        
        if transition_type == 'fade':
            clip1 = clip1.fx(vfx.fadeout, duration)
            clip2 = clip2.fx(vfx.fadein, duration)
            return concatenate_videoclips([clip1, clip2], method="compose")
            
        elif transition_type == 'wipe':
            # Реализация wipe-перехода
            def make_frame(t):
                if t < duration:
                    # Линейная интерполяция между кадрами
                    t1 = clip1.duration - duration + t
                    t2 = t
                    frame1 = clip1.get_frame(t1)
                    frame2 = clip2.get_frame(t2)
                    
                    # Создание маски для wipe-эффекта (слева направо)
                    h, w = frame1.shape[:2]
                    mask = np.zeros((h, w))
                    wipe_width = int(w * t / duration)
                    mask[:, :wipe_width] = 1
                    
                    # Применение маски
                    result = frame1.copy()
                    for i in range(h):
                        for j in range(w):
                            if j < wipe_width:
                                result[i, j] = frame2[i, j]
                                
                    return result
                else:
                    return clip2.get_frame(t - duration)
                    
            # Создаем новый клип с обработкой кадров
            new_clip = VideoFileClip(make_frame, duration=clip1.duration + clip2.duration - duration)
            return new_clip
            
        elif transition_type == 'slide':
            # Упрощенная реализация slide-перехода
            from moviepy.editor import CompositeVideoClip
            
            clip1 = clip1.set_position(("center", "center"))
            clip2 = clip2.set_position(lambda t: ('center', 'center') if t > duration else (int(clip1.w * (t/duration - 1)), 'center'))
            
            # Создаем композитный клип
            comp_clip = CompositeVideoClip([clip1, clip2.set_start(clip1.duration - duration)])
            return comp_clip
            
        # По умолчанию склеиваем без перехода
        return concatenate_videoclips([clip1, clip2])
        
    def apply_transition_to_files(self, video_path1: str, video_path2: str, 
                                 transition_type: str, duration: float = 1.0,
                                 output_path: Optional[str] = None) -> str:
        """
        Применяет переход между двумя видеофайлами с использованием ffmpeg
        
        Args:
            video_path1: путь к первому видео
            video_path2: путь к второму видео
            transition_type: тип перехода
            duration: длительность перехода
            output_path: путь для сохранения результата
            
        Returns:
            путь к видео с примененным переходом
        """
        # Проверяем наличие файлов
        if not os.path.exists(video_path1) or not os.path.exists(video_path2):
            raise FileNotFoundError(f"Видеофайлы не найдены: {video_path1}, {video_path2}")
            
        # Если выходной путь не указан, создаем временный
        if output_path is None:
            output_filename = f"{uuid.uuid4()}_transition.mp4"
            output_path = os.path.join(self.temp_dir, output_filename)
            
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
        # Если можем использовать прямой вызов ffmpeg и переход поддерживается
        if self.use_ffmpeg_direct and transition_type in ['fade', 'wipe']:
            try:
                # Получаем информацию о видео
                info1 = VideoUtils.get_video_info(video_path1)
                
                if transition_type == 'fade':
                    # Создаем временные файлы для обработанных видео
                    temp_fade_out = os.path.join(self.temp_dir, f"{uuid.uuid4()}_fade_out.mp4")
                    temp_fade_in = os.path.join(self.temp_dir, f"{uuid.uuid4()}_fade_in.mp4")
                    
                    try:
                        # Применяем fade out к первому видео
                        (
                            ffmpeg
                            .input(video_path1)
                            .filter('fade', type='out', start_time=info1['duration']-duration, duration=duration)
                            .output(temp_fade_out, **{'c:a': 'copy'})
                            .run(quiet=True, overwrite_output=True)
                        )
                        
                        # Применяем fade in ко второму видео
                        (
                            ffmpeg
                            .input(video_path2)
                            .filter('fade', type='in', start_time=0, duration=duration)
                            .output(temp_fade_in, **{'c:a': 'copy'})
                            .run(quiet=True, overwrite_output=True)
                        )
                        
                        # Создаем файл списка для конкатенации
                        concat_list = os.path.join(self.temp_dir, f"{uuid.uuid4()}_concat.txt")
                        with open(concat_list, 'w') as f:
                            f.write(f"file '{os.path.abspath(temp_fade_out)}'\n")
                            f.write(f"file '{os.path.abspath(temp_fade_in)}'\n")
                            
                        # Склеиваем видео
                        (
                            ffmpeg
                            .input(concat_list, format='concat', safe=0)
                            .output(output_path, c='copy')
                            .run(quiet=True, overwrite_output=True)
                        )
                        
                        return output_path
                        
                    finally:
                        # Удаляем временные файлы
                        for temp_file in [temp_fade_out, temp_fade_in, concat_list]:
                            if os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except:
                                    pass
                                    
                elif transition_type == 'wipe':
                    # Вырезаем последние секунды первого видео
                    temp_last_part = os.path.join(self.temp_dir, f"{uuid.uuid4()}_last_part.mp4")
                    temp_first_part = os.path.join(self.temp_dir, f"{uuid.uuid4()}_first_part.mp4")
                    
                    # Создаем комплексный переход через xfade
                    (
                        ffmpeg
                        .input(video_path1)
                        .input(video_path2)
                        .filter_multi_output('xfade', transition='wipeleft', duration=duration, offset=info1['duration']-duration)
                        .output(output_path)
                        .run(quiet=True, overwrite_output=True)
                    )
                    
                    return output_path
                    
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при применении перехода {transition_type} через ffmpeg: {e}. Переключаемся на MoviePy.")
        
        # Резервный вариант через MoviePy
        try:
            clip1 = VideoFileClip(video_path1)
            clip2 = VideoFileClip(video_path2)
            
            processed_clip = self.apply_transition(clip1, clip2, transition_type, duration)
            processed_clip.write_videofile(output_path, codec='libx264')
            
            clip1.close()
            clip2.close()
            processed_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при применении перехода {transition_type}: {e}")
            
            # В случае ошибки просто склеиваем видео без перехода
            concat_list = os.path.join(self.temp_dir, f"{uuid.uuid4()}_concat_simple.txt")
            
            try:
                with open(concat_list, 'w') as f:
                    f.write(f"file '{os.path.abspath(video_path1)}'\n")
                    f.write(f"file '{os.path.abspath(video_path2)}'\n")
                    
                (
                    ffmpeg
                    .input(concat_list, format='concat', safe=0)
                    .output(output_path, c='copy')
                    .run(quiet=True, overwrite_output=True)
                )
                
                return output_path
                
            except Exception as e2:
                logger.error(f"Ошибка при простом склеивании видео: {e2}")
                return video_path1
            finally:
                if os.path.exists(concat_list):
                    try:
                        os.remove(concat_list)
                    except:
                        pass 