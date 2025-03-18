# Скрипт для запуска Docker-контейнеров
# Использование: 
# .\scripts\docker-run.ps1 [train|api|bash]

param (
    [string]$mode = "bash"  # По умолчанию запускаем интерактивную оболочку
)

# Проверка наличия Docker
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerInstalled) {
    Write-Error "Docker не установлен. Пожалуйста, установите Docker Desktop."
    exit 1
}

# Проверка, что Docker запущен
$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Write-Error "Docker не запущен. Пожалуйста, запустите Docker Desktop."
    exit 1
}

# Создаем каталоги, если их нет
if (-not (Test-Path "backend/ai_service/trained_models")) {
    New-Item -ItemType Directory -Path "backend/ai_service/trained_models" -Force | Out-Null
    Write-Host "Создана директория для моделей: backend/ai_service/trained_models"
}

# Проверяем наличие сборки образа
$imageExists = docker images -q video-copier 2>$null
if (-not $imageExists) {
    Write-Host "Образ Docker не найден. Выполняется сборка..."
    docker-compose build
}

# Запускаем контейнер в зависимости от выбранного режима
switch ($mode) {
    "train" {
        Write-Host "Запуск обучения моделей..."
        docker-compose run train
    }
    "api" {
        Write-Host "Запуск API-сервера на порту 5000..."
        docker-compose up api
    }
    "bash" {
        Write-Host "Запуск интерактивной оболочки bash..."
        docker-compose run tensorflow bash
    }
    default {
        Write-Host "Неизвестный режим '$mode'. Используйте 'train', 'api' или 'bash'."
        exit 1
    }
}

Write-Host "Готово!" 