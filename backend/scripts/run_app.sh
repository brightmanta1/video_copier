#!/bin/bash

echo "Starting Video Copier Backend..."

# Переход в директорию проекта
cd "$(dirname "$0")/.."

# Проверка наличия venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r python/requirements.txt
else
    source venv/bin/activate
fi

# Установка переменных окружения
export AVE_PATH="../datasets/AVE"
export EDIT3K_PATH="../datasets/Edit3K"
export OUTPUT_DIR="../output"

# Проверка директорий
if [ ! -d "$AVE_PATH" ]; then
    echo "WARNING: AVE dataset not found at $AVE_PATH"
fi

if [ ! -d "$EDIT3K_PATH" ]; then
    echo "WARNING: Edit3K dataset not found at $EDIT3K_PATH"
fi

# Создание директории для результатов
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Запуск Flask-приложения
echo "Starting Flask server..."
python python/app/app.py 