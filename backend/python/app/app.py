from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Конфигурация логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api')

def create_app():
    # Создаем экземпляр Flask
    app = Flask(__name__)
    
    # Загружаем конфигурацию
    app.config.from_object('app.config.AppConfig')
    
    # Настраиваем CORS
    cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})
    
    # Настраиваем обработчики запросов
    from app.api import routes
    app.register_blueprint(routes.api_blueprint)
    
    # Обработчик ошибок 404
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested URL was not found on the server.'
        }), 404
    
    # Обработчик ошибок 500
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred on the server.'
        }), 500
    
    # Проверяем окружение Vercel
    is_vercel = os.environ.get('VERCEL', False)
    if is_vercel:
        logger.info(f"Running on Vercel environment: {os.environ.get('VERCEL_ENV')}")
    
    return app

# Если запускается напрямую
if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get('API_PORT', 5000))
    host = os.environ.get('API_HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug) 