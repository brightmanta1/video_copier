from flask import Flask, jsonify, request
import json
import os
import sys

# Добавляем родительский каталог в путь импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем наше приложение
from app.app import create_app

# Создаем экземпляр приложения для Vercel
flask_app = create_app()

# Настраиваем CORS
@flask_app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# Обработчик для проверки состояния API
@flask_app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Video Copier API is running'
    })

# Обработчик для сбора информации о системе
@flask_app.route('/api/system-info', methods=['GET'])
def system_info():
    return jsonify({
        'python_version': sys.version,
        'environment': os.environ.get('VERCEL_ENV', 'development'),
        'region': os.environ.get('VERCEL_REGION', 'unknown')
    })

# Точка входа для Vercel Serverless Functions
app = flask_app 