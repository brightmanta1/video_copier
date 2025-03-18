@echo off
echo Запуск TensorFlow в Docker...
echo.
echo Пожалуйста, убедитесь что Docker Desktop установлен и запущен!
echo.
timeout /t 3 >nul

REM Проверяем, запущен ли Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Docker не запущен. Пожалуйста, запустите Docker Desktop.
    pause
    exit /b 1
)

REM Проверяем существование образа
docker images | findstr "tensorflow/tensorflow" >nul
if errorlevel 1 (
    echo Образ TensorFlow не найден, загружаем из Docker Hub...
    docker pull tensorflow/tensorflow:2.12.0-gpu
    if errorlevel 1 (
        echo Ошибка: Не удалось загрузить образ TensorFlow.
        echo Пробуем загрузить версию без GPU...
        docker pull tensorflow/tensorflow:2.12.0
        if errorlevel 1 (
            echo Ошибка: Не удалось загрузить образ TensorFlow.
            pause
            exit /b 1
        )
    )
)

REM Проверяем, запущен ли контейнер
for /f %%i in ('docker ps -q -f "ancestor=tensorflow/tensorflow"') do set CONTAINER_ID=%%i
if defined CONTAINER_ID (
    echo Контейнер TensorFlow уже запущен с ID: %CONTAINER_ID%
) else (
    echo Запуск нового контейнера TensorFlow...
    docker run -d --rm -v "%cd%:/app" -p 8501:8501 tensorflow/tensorflow:2.12.0-gpu python -c "import time; import tensorflow as tf; print(f'TensorFlow {tf.__version__} готов к использованию'); time.sleep(3600*24)"
    if errorlevel 1 (
        echo Ошибка при запуске GPU-версии. Пробуем версию без GPU...
        docker run -d --rm -v "%cd%:/app" -p 8501:8501 tensorflow/tensorflow:2.12.0 python -c "import time; import tensorflow as tf; print(f'TensorFlow {tf.__version__} готов к использованию'); time.sleep(3600*24)"
    )
    for /f %%i in ('docker ps -q -f "ancestor=tensorflow/tensorflow"') do set CONTAINER_ID=%%i
    if defined CONTAINER_ID (
        echo Контейнер TensorFlow успешно запущен с ID: %CONTAINER_ID%
    ) else (
        echo Ошибка: Не удалось запустить контейнер TensorFlow.
        pause
        exit /b 1
    )
)

echo.
echo TensorFlow в Docker успешно запущен и готов к использованию!
echo Контейнер будет работать в фоновом режиме и автоматически использоваться вашим приложением.
echo.
echo Для остановки контейнера выполните: docker stop %CONTAINER_ID%
echo.
pause 