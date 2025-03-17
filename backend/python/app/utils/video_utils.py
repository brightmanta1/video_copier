import cv2
import numpy as np
import os
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import logging

# Импортируем ffmpeg-python без сложной обработки исключений
# ffmpeg-python уже установлен, пакет ffmpeg удален
import ffmpeg

from pathlib import Path
from moviepy.editor import VideoFileClip

logger = logging.getLogger('VideoUtils')

class VideoUtils:
    """
    Утилиты для обработки видео:
    - Извлечение метаданных видео
    - Генерация превью и миниатюр
    - Валидация форматов
    - Оптимизированное чтение/запись
    - Прямые вызовы ffmpeg для высокопроизводительных операций
    """
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Получает информацию о видеофайле
        
        Args:
            video_path: путь к видео
            
        Returns:
            словарь с информацией о видео
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
        
        try:
            # Используем ffmpeg-python для получения метаданных (быстрее MoviePy)
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                              if stream['codec_type'] == 'video'), None)
            
            if video_stream is None:
                raise ValueError(f"Не удалось найти видео поток в файле: {video_path}")
                
            # Извлекаем данные из результата ffprobe
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Получаем продолжительность видео
            if 'duration' in video_stream:
                duration = float(video_stream['duration'])
            else:
                duration = float(probe['format']['duration'])
                
            # Получаем частоту кадров
            fps_parts = video_stream.get('r_frame_rate', '').split('/')
            if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                fps = float(int(fps_parts[0]) / int(fps_parts[1]))
            else:
                fps = float(video_stream.get('avg_frame_rate', '30/1').split('/')[0])
                
            # Рассчитываем количество кадров
            frame_count = int(duration * fps)
            
            # Проверяем аудио поток
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            if audio_stream:
                audio_info = {
                    "has_audio": True,
                    "audio_codec": audio_stream.get('codec_name', 'unknown'),
                    "audio_channels": int(audio_stream.get('channels', 0)),
                    "audio_sample_rate": int(audio_stream.get('sample_rate', 0))
                }
            else:
                audio_info = {"has_audio": False, "audio_codec": None, "audio_channels": 0, "audio_sample_rate": 0}
            
        except ffmpeg.Error as e:
            logger.warning(f"Ошибка ffmpeg при получении информации о видео: {e}")
            
            # Запасной вариант - используем OpenCV
            return VideoUtils._get_video_info_opencv(video_path)
            
        # Формируем результат
        return {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "filesize_mb": os.path.getsize(video_path) / (1024 * 1024),
            "audio": audio_info,
            "codec": video_stream.get('codec_name', 'unknown'),
            "bitrate": int(probe['format'].get('bit_rate', 0)),
            "hash": VideoUtils.compute_file_hash(video_path)
        }
    
    @staticmethod
    def _get_video_info_opencv(video_path: str) -> Dict[str, Any]:
        """
        Резервный метод получения информации о видео через OpenCV
        
        Args:
            video_path: путь к видео
            
        Returns:
            словарь с информацией о видео
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        # Получаем базовую информацию
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Освобождаем ресурсы
        cap.release()
        
        # Дополнительная информация через MoviePy
        try:
            clip = VideoFileClip(video_path)
            audio_info = {
                "has_audio": clip.audio is not None,
                "audio_codec": "unknown",
                "audio_channels": clip.audio.nchannels if clip.audio else 0,
                "audio_sample_rate": clip.audio.fps if clip.audio else 0
            }
            clip.close()
        except Exception as e:
            logger.warning(f"Ошибка при получении аудио-информации: {e}")
            audio_info = {"has_audio": False, "audio_codec": None, "audio_channels": 0, "audio_sample_rate": 0}
        
        # Формируем результат
        return {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "filesize_mb": os.path.getsize(video_path) / (1024 * 1024),
            "audio": audio_info,
            "codec": "unknown",
            "bitrate": 0,
            "hash": VideoUtils.compute_file_hash(video_path)
        }
    
    @staticmethod
    def generate_thumbnail(video_path: str, output_path: str = None, 
                          time_percent: float = 0.2, width: int = 320) -> str:
        """
        Генерирует миниатюру видео
        
        Args:
            video_path: путь к видео
            output_path: путь для сохранения миниатюры
            time_percent: процент времени видео для миниатюры (0.0-1.0)
            width: ширина миниатюры
            
        Returns:
            путь к миниатюре
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
        # Если путь не указан, создаем в том же каталоге
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_thumb.jpg"
            
        # Проверяем, существует ли каталог для сохранения
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        # Получаем кадр для миниатюры
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(frame_count * time_percent)
        
        # Устанавливаем позицию
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            # В случае ошибки берем первый кадр
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Не удалось получить кадр из видео: {video_path}")
        
        # Изменяем размер
        height, width_orig, _ = frame.shape
        new_height = int(height * width / width_orig)
        frame_resized = cv2.resize(frame, (width, new_height))
        
        # Сохраняем миниатюру
        cv2.imwrite(output_path, frame_resized)
        
        # Освобождаем ресурсы
        cap.release()
        
        return output_path
    
    @staticmethod
    def generate_preview_frames(video_path: str, output_dir: str = None, 
                               frame_count: int = 5, width: int = 320) -> List[str]:
        """
        Генерирует кадры-превью видео
        
        Args:
            video_path: путь к видео
            output_dir: директория для сохранения кадров
            frame_count: количество кадров
            width: ширина кадров
            
        Returns:
            список путей к кадрам
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
        # Если каталог не указан, создаем подкаталог
        if output_dir is None:
            base_path = os.path.splitext(video_path)[0]
            output_dir = f"{base_path}_frames"
            
        # Создаем каталог, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        # Получаем общее количество кадров
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Определяем интервал между кадрами
        if total_frames <= frame_count:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * total_frames / frame_count) for i in range(frame_count)]
            
        result_paths = []
        
        for i, frame_idx in enumerate(frame_indices):
            # Устанавливаем позицию
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Изменяем размер
            height, width_orig, _ = frame.shape
            new_height = int(height * width / width_orig)
            frame_resized = cv2.resize(frame, (width, new_height))
            
            # Сохраняем кадр
            output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame_resized)
            result_paths.append(output_path)
            
        # Освобождаем ресурсы
        cap.release()
        
        return result_paths
    
    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = 'md5', buffer_size: int = 65536) -> str:
        """
        Вычисляет хеш файла
        
        Args:
            file_path: путь к файлу
            algorithm: алгоритм хеширования ('md5', 'sha1', 'sha256')
            buffer_size: размер буфера для чтения
            
        Returns:
            хеш файла
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
            
        # Выбираем алгоритм
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
            
        # Вычисляем хеш
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash_obj.update(data)
                
        return hash_obj.hexdigest()
    
    @staticmethod
    def is_video_valid(video_path: str) -> bool:
        """
        Проверяет, является ли файл валидным видео
        
        Args:
            video_path: путь к видео
            
        Returns:
            True если видео корректное, иначе False
        """
        if not os.path.exists(video_path):
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Пытаемся получить первый кадр
        ret, frame = cap.read()
        cap.release()
        
        return ret
    
    @staticmethod
    def get_optimal_segment_size(duration: float) -> int:
        """
        Определяет оптимальный размер сегмента для параллельной обработки
        
        Args:
            duration: длительность видео в секундах
            
        Returns:
            размер сегмента в секундах
        """
        if duration <= 10:
            return int(duration)
        elif duration <= 60:
            return 10
        elif duration <= 300:  # 5 минут
            return 30
        else:
            return 60  # Для длинных видео используем минутные сегменты
    
    @staticmethod
    def convert_to_mp4(input_path: str, output_path: Optional[str] = None, 
                      use_ffmpeg_direct: bool = True, quality: str = "medium") -> str:
        """
        Конвертирует видео в MP4 формат с H.264 кодеком
        
        Args:
            input_path: путь к исходному видео
            output_path: путь для сохранения (опционально)
            use_ffmpeg_direct: использовать прямой вызов ffmpeg (быстрее)
            quality: качество (низкое, среднее, высокое)
            
        Returns:
            путь к конвертированному видео
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Файл не найден: {input_path}")
            
        # Если выходной путь не указан, создаем автоматически
        if output_path is None:
            base_path = os.path.splitext(input_path)[0]
            output_path = f"{base_path}_converted.mp4"
            
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
        # Конвертируем видео с использованием ffmpeg-python (для скорости)
        if use_ffmpeg_direct:
            try:
                # Устанавливаем параметры кодирования в зависимости от качества
                if quality == "low":
                    preset = "veryfast"
                    crf = 28
                elif quality == "high":
                    preset = "slow"
                    crf = 18
                else:  # medium
                    preset = "medium"
                    crf = 23
                    
                # Запускаем конвертацию
                (
                    ffmpeg
                    .input(input_path)
                    .output(output_path, 
                           **{'c:v': 'libx264', 
                              'preset': preset, 
                              'crf': str(crf), 
                              'c:a': 'aac'})
                    .run(quiet=True, overwrite_output=True)
                )
                return output_path
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при прямом вызове ffmpeg: {e}. Переключаемся на MoviePy.")
                
        # Резервный вариант через MoviePy
        try:
            video = VideoFileClip(input_path)
            video.write_videofile(output_path, codec='libx264')
            video.close()
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при конвертировании видео: {e}")
            raise
            
    @staticmethod
    def extract_audio(video_path: str, output_path: Optional[str] = None, 
                     format: str = "mp3", use_ffmpeg_direct: bool = True) -> str:
        """
        Извлекает аудио из видео
        
        Args:
            video_path: путь к видео
            output_path: путь для сохранения аудио (опционально)
            format: формат аудио (mp3, wav)
            use_ffmpeg_direct: использовать прямой вызов ffmpeg (быстрее)
            
        Returns:
            путь к аудиофайлу
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Файл не найден: {video_path}")
            
        # Если выходной путь не указан, создаем автоматически
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_audio.{format}"
            
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Извлекаем аудио с использованием ffmpeg-python (быстрее)
        if use_ffmpeg_direct:
            try:
                audio_codec = 'libmp3lame' if format == 'mp3' else 'pcm_s16le'
                
                (
                    ffmpeg
                    .input(video_path)
                    .output(output_path, **{'c:a': audio_codec, 'q:a': '2'})
                    .run(quiet=True, overwrite_output=True)
                )
                return output_path
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при прямом вызове ffmpeg: {e}. Переключаемся на MoviePy.")
                
        # Резервный вариант через MoviePy
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                raise ValueError("Видео не содержит аудиодорожки")
                
            video.audio.write_audiofile(output_path)
            video.close()
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудио: {e}")
            raise
            
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, fps: Optional[float] = None, 
                      use_ffmpeg_direct: bool = True, quality: int = 95) -> List[str]:
        """
        Извлекает кадры из видео с высокой производительностью
        
        Args:
            video_path: путь к видео
            output_dir: директория для сохранения кадров
            fps: частота извлечения кадров (если None, то извлекаются все кадры)
            use_ffmpeg_direct: использовать прямой вызов ffmpeg (быстрее)
            quality: качество JPEG (1-100)
            
        Returns:
            список путей к извлеченным кадрам
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Файл не найден: {video_path}")
            
        # Создаем директорию, если не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем информацию о видео
        video_info = VideoUtils.get_video_info(video_path)
        
        # Если fps не указан, используем fps видео
        if fps is None:
            fps = video_info["fps"]
            
        # Используем ffmpeg для быстрого извлечения кадров
        if use_ffmpeg_direct:
            try:
                output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
                
                # Настраиваем параметры извлечения кадров
                if fps < video_info["fps"]:
                    # Извлекаем с указанной частотой
                    (
                        ffmpeg
                        .input(video_path)
                        .filter('fps', fps=fps)
                        .output(output_pattern, **{'q:v': str(int(quality * 31 / 100))})
                        .run(quiet=True, overwrite_output=True)
                    )
                else:
                    # Извлекаем все кадры
                    (
                        ffmpeg
                        .input(video_path)
                        .output(output_pattern, **{'q:v': str(int(quality * 31 / 100))})
                        .run(quiet=True, overwrite_output=True)
                    )
                
                # Получаем список созданных файлов
                frame_files = sorted([
                    os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                    if f.startswith("frame_") and f.endswith(".jpg")
                ])
                
                return frame_files
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при прямом вызове ffmpeg: {e}. Переключаемся на OpenCV.")
        
        # Резервный вариант с использованием OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        frame_files = []
        frame_count = 0
        
        # Расчёт шага для указанного fps
        step = 1
        if fps < video_info["fps"]:
            step = round(video_info["fps"] / fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Пропускаем кадры в зависимости от желаемого fps
            if frame_count % step == 0:
                output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                # Устанавливаем параметры качества JPEG
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                cv2.imwrite(output_path, frame, encode_params)
                frame_files.append(output_path)
                
            frame_count += 1
            
        cap.release()
        return frame_files
        
    @staticmethod
    def cut_video_segment(input_path: str, output_path: str, start_time: float, 
                         end_time: float, use_ffmpeg_direct: bool = True) -> str:
        """
        Вырезает сегмент видео
        
        Args:
            input_path: путь к исходному видео
            output_path: путь для сохранения
            start_time: время начала (в секундах)
            end_time: время окончания (в секундах)
            use_ffmpeg_direct: использовать прямой вызов ffmpeg (быстрее)
            
        Returns:
            путь к сегменту видео
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Файл не найден: {input_path}")
            
        # Проверяем, что временные отметки корректны
        if start_time < 0:
            start_time = 0
            
        if end_time <= start_time:
            raise ValueError("Время окончания должно быть больше времени начала")
            
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
        # Вырезаем сегмент с использованием ffmpeg
        if use_ffmpeg_direct:
            try:
                (
                    ffmpeg
                    .input(input_path, ss=start_time, to=end_time)
                    .output(output_path, c='copy')
                    .run(quiet=True, overwrite_output=True)
                )
                return output_path
            except ffmpeg.Error as e:
                logger.warning(f"Ошибка при прямом вызове ffmpeg: {e}. Переключаемся на MoviePy.")
                
        # Резервный вариант через MoviePy
        try:
            video = VideoFileClip(input_path)
            segment = video.subclip(start_time, end_time)
            segment.write_videofile(output_path)
            video.close()
            segment.close()
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при вырезании сегмента видео: {e}")
            raise 