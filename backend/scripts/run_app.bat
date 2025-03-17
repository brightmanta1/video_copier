@echo off
echo Starting Video Copier Backend...

REM Переход в директорию проекта
cd /d %~dp0\..

REM Проверка наличия venv
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install -r python\requirements.txt
) else (
    call venv\Scripts\activate
)

REM Установка переменных окружения
set AVE_PATH=..\datasets\AVE
set EDIT3K_PATH=..\datasets\Edit3K
set OUTPUT_DIR=..\output

REM Проверка директорий
if not exist %AVE_PATH% (
    echo WARNING: AVE dataset not found at %AVE_PATH%
)

if not exist %EDIT3K_PATH% (
    echo WARNING: Edit3K dataset not found at %EDIT3K_PATH%
)

REM Создание директории для результатов
if not exist %OUTPUT_DIR% (
    mkdir %OUTPUT_DIR%
)

REM Запуск Flask-приложения
echo Starting Flask server...
python python\app\app.py

pause 