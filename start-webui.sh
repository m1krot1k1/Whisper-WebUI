#!/bin/bash

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем переменные окружения для Hugging Face
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_TOKEN="${HF_TOKEN:-}"

# Устанавливаем переменные окружения для аутентификации
export WHISPER_USERNAME="${WHISPER_USERNAME:-@dmin4}"
export WHISPER_PASSWORD="${WHISPER_PASSWORD:-b#*4#3xT0B3*Rn4g}"
export WHISPER_THEME="${WHISPER_THEME:-dark}"

# Запускаем приложение на 0.0.0.0 с портом 7860
# Используем модель Systran, которая совместима с faster-whisper
# Добавляем аутентификацию и темную тему
python app.py --server_name 0.0.0.0 --server_port 7860 --username "$WHISPER_USERNAME" --password "$WHISPER_PASSWORD" --theme "$WHISPER_THEME"

echo "Приложение запущено на http://0.0.0.0:7860"
