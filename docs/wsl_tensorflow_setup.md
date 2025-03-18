# Настройка TensorFlow в WSL для Video Copier

Это руководство поможет настроить среду разработки для проекта Video Copier в WSL с TensorFlow.

## Шаг 1: Установка WSL и Ubuntu

1. Откройте PowerShell с правами администратора и выполните:
   ```powershell
   wsl --install -d Ubuntu
   ```

2. Следуйте инструкциям на экране для создания пользователя и пароля
3. После завершения установки, WSL Ubuntu будет готова к использованию

## Шаг 2: Настройка Python и зависимостей в WSL

1. Откройте терминал WSL (Ubuntu) и выполните следующие команды:

   ```bash
   # Обновите пакеты
   sudo apt update && sudo apt upgrade -y

   # Установите необходимые инструменты разработки
   sudo apt install -y build-essential libssl-dev libffi-dev python3-dev git

   # Установите Python 3.10 (для TensorFlow)
   sudo apt install python3.10 python3.10-dev python3.10-venv python3-pip -y

   # Создайте виртуальное окружение с Python 3.10
   python3.10 -m venv ~/video-copier-env

   # Активируйте виртуальное окружение
   source ~/video-copier-env/bin/activate
   ```

## Шаг 3: Клонирование проекта и установка зависимостей

1. Клонируйте репозиторий:
   ```bash
   cd ~
   git clone https://github.com/yourusername/video-copier.git
   cd video-copier
   ```

2. Установите зависимости AI-сервиса:
   ```bash
   # Активируйте виртуальное окружение, если еще не активировано
   source ~/video-copier-env/bin/activate

   # Установите TensorFlow и другие зависимости
   pip install tensorflow numpy opencv-python-headless scikit-learn scipy matplotlib pillow h5py tqdm gitpython psutil
   ```

## Шаг 4: Настройка доступа к GPU (опционально)

Если у вас есть GPU NVIDIA, вы можете настроить TensorFlow для использования GPU:

1. Установите драйверы NVIDIA в Windows (если еще не установлены)
2. Установите CUDA и cuDNN в WSL:
   ```bash
   # Установка CUDA Toolkit
   sudo apt install -y nvidia-cuda-toolkit

   # Проверка установки
   nvcc --version
   ```

## Шаг 5: Обучение моделей

1. Запустите обучение моделей:
   ```bash
   # Активируйте виртуальное окружение
   source ~/video-copier-env/bin/activate

   # Перейдите в директорию проекта
   cd ~/video-copier

   # Запустите скрипт обучения
   python backend/ai_service/run_training.py
   ```

## Интеграция с проектом на Windows

Для использования моделей, обученных в WSL, с проектом на Windows:

1. Скопируйте обученные модели из WSL в ваш проект на Windows:
   ```bash
   # В WSL
   cp ~/video-copier/backend/ai_service/trained_models/*.h5 /mnt/c/path/to/your/windows/project/backend/ai_service/trained_models/
   ```

2. В файле конфигурации на Windows укажите путь к обученным моделям

## Решение проблем

### TensorFlow не устанавливается

Убедитесь, что вы используете Python 3.10 или более раннюю версию, так как TensorFlow не поддерживает Python 3.11+ на момент создания этого руководства.

### Ошибки с CUDA и GPU

Если у вас возникают проблемы с использованием GPU:
1. Проверьте совместимость версий CUDA, cuDNN и TensorFlow
2. Убедитесь, что драйверы NVIDIA правильно установлены в Windows
3. Используйте `tf.config.list_physical_devices('GPU')` для проверки видимости GPU

### Проблемы с памятью

При обучении больших моделей на WSL можно столкнуться с ограничениями памяти:
1. Увеличьте размер памяти, выделенной для WSL в файле `.wslconfig`
2. Уменьшите размер батча (`batch_size`) при обучении
3. Используйте модели с меньшим количеством параметров 