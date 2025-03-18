#!/bin/bash
# Скрипт для запуска Docker-контейнеров
# Использование: 
# ./scripts/docker-run.sh [train|api|bash]

MODE=${1:-bash}  # По умолчанию запускаем интерактивную оболочку

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "Docker не установлен. Пожалуйста, установите Docker."
    exit 1
fi

# Проверка, что Docker запущен
if ! docker info &> /dev/null; then
    echo "Docker не запущен. Пожалуйста, запустите Docker."
    exit 1
fi

# Создаем каталоги, если их нет
if [ ! -d "backend/ai_service/trained_models" ]; then
    mkdir -p backend/ai_service/trained_models
    echo "Создана директория для моделей: backend/ai_service/trained_models"
fi

# Проверяем наличие сборки образа
if [[ -z $(docker images -q video-copier 2>/dev/null) ]]; then
    echo "Образ Docker не найден. Выполняется сборка..."
    docker-compose build
fi

# Запускаем контейнер в зависимости от выбранного режима
case $MODE in
    train)
        echo "Запуск обучения моделей..."
        docker-compose run train
        ;;
    api)
        echo "Запуск API-сервера на порту 5000..."
        docker-compose up api
        ;;
    bash)
        echo "Запуск интерактивной оболочки bash..."
        docker-compose run tensorflow bash
        ;;
    *)
        echo "Неизвестный режим '$MODE'. Используйте 'train', 'api' или 'bash'."
        exit 1
        ;;
esac

echo "Готово!" 