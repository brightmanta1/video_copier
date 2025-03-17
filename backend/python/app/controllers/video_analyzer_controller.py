from flask import request, jsonify, Blueprint, send_from_directory, current_app
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
import json

logger = logging.getLogger('VideoAnalyzerController')

# Импорт сервиса с обработкой исключений
try:
    from ..services.video_analyzer_service import VideoAnalyzerService
    service_import_error = None
except ImportError as e:
    service_import_error = str(e)
    logger.error(f"Ошибка импорта VideoAnalyzerService: {e}")
    VideoAnalyzerService = None

video_analyzer_bp = Blueprint('video_analyzer', __name__)

# Инициализация сервиса с проверкой
if VideoAnalyzerService is not None:
    try:
        video_analyzer_service = VideoAnalyzerService()
        logger.info("VideoAnalyzerService успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации VideoAnalyzerService: {e}")
        video_analyzer_service = None
else:
    video_analyzer_service = None
    logger.error("VideoAnalyzerService недоступен из-за ошибки импорта")

@video_analyzer_bp.route('/analyze', methods=['POST'])
def analyze_video():
    """Анализ загруженного видео"""
    # Проверка работоспособности сервиса
    if video_analyzer_service is None:
        logger.error("Невозможно выполнить анализ: сервис не инициализирован")
        return jsonify({'error': 'Service unavailable'}), 503
        
    # Проверка наличия файла
    if 'video' not in request.files:
        logger.warning("Запрос на анализ без видеофайла")
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if not video_file.filename:
        logger.warning("Запрос на анализ с пустым именем файла")
        return jsonify({'error': 'Empty filename'}), 400
        
    filename = secure_filename(video_file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    logger.info(f"Начало анализа видео: {filename}")
    
    try:
        video_file.save(temp_path)
        
        # Анализ видео
        analysis = video_analyzer_service.analyze_video(temp_path)
        logger.info(f"Анализ видео {filename} завершен успешно")
        
        # Очистка временного файла
        os.remove(temp_path)
        os.rmdir(temp_dir)
        return jsonify(analysis)
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Ошибка при анализе видео {filename}: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Гарантированная очистка временных файлов
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass

@video_analyzer_bp.route('/apply-structure', methods=['POST'])
def apply_structure():
    """Применение структуры редактирования к новому видео"""
    # Проверка работоспособности сервиса
    if video_analyzer_service is None:
        logger.error("Невозможно применить структуру: сервис не инициализирован")
        return jsonify({'error': 'Service unavailable'}), 503
        
    # Проверка наличия файла и структуры
    if 'video' not in request.files:
        logger.warning("Запрос без видеофайла")
        return jsonify({'error': 'No target video provided'}), 400
    
    if 'structure' not in request.form:
        logger.warning("Запрос без структуры редактирования")
        return jsonify({'error': 'No editing structure provided'}), 400
    
    video_file = request.files['video']
    if not video_file.filename:
        logger.warning("Запрос с пустым именем файла")
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        structure = json.loads(request.form['structure'])
    except json.JSONDecodeError as e:
        logger.error(f"Неверный формат JSON структуры: {e}")
        return jsonify({'error': 'Invalid JSON structure format'}), 400
    
    filename = secure_filename(video_file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    logger.info(f"Начало применения структуры к видео: {filename}")
    
    try:
        video_file.save(temp_path)
        
        # Применение структуры
        output_path = video_analyzer_service.apply_structure(temp_path, structure)
        
        # Формируем URL для доступа к результату
        api_prefix = current_app.config.get('API_PREFIX', '/api')
        result_url = f"{api_prefix}/results/{os.path.basename(output_path)}"
        
        logger.info(f"Структура успешно применена к видео {filename}")
        
        # Очистка временного файла
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return jsonify({'result_url': result_url})
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Ошибка при применении структуры к видео {filename}: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Гарантированная очистка временных файлов
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass

@video_analyzer_bp.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    """Получение результирующего видео"""
    # Проверка работоспособности сервиса
    if video_analyzer_service is None:
        logger.error("Невозможно получить результат: сервис не инициализирован")
        return jsonify({'error': 'Service unavailable'}), 503
        
    try:
        output_dir = video_analyzer_service.get_output_dir()
        logger.info(f"Запрос результата: {filename}")
        
        # Проверка существования файла
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Запрошенный файл не найден: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        return send_from_directory(output_dir, filename)
    except Exception as e:
        logger.error(f"Ошибка при получении файла {filename}: {e}")
        return jsonify({'error': str(e)}), 500 