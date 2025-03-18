import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь для импорта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.python.app.utils.tensorflow_docker import TensorFlowDocker

def main():
    tf_docker = TensorFlowDocker()
    print(f'TensorFlow доступен: {tf_docker.is_tensorflow_available()}')

if __name__ == '__main__':
    main() 