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
docker images | findstr "tensorflow" >nul
if errorlevel 1 (
    echo Образ TensorFlow не найден, выполняется сборка...
    docker build -t tensorflow_app .
)

REM Запускаем контейнер
echo Запуск TensorFlow контейнера...
echo.
docker run -it --rm -v "%cd%:/app" tensorflow_app /bin/bash

echo.
echo Сеанс Docker завершен.
pause 