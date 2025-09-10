#!/bin/bash

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем переменные окружения для Hugging Face
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_TOKEN="${HF_TOKEN:-}"

# Запускаем приложение на 0.0.0.0 с портом 7860
# Используем модель Systran, которая совместима с faster-whisper
python app.py --server_name 0.0.0.0 --server_port 7860

echo "Приложение запущено на http://0.0.0.0:7860"
