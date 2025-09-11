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

# Функция для завершения всех процессов
cleanup() {
    echo "Завершение работы..."
    kill $API_PID $WEBUI_PID 2>/dev/null
    exit 0
}

# Устанавливаем обработчик сигналов
trap cleanup SIGINT SIGTERM

# Запускаем API сервер в фоновом режиме
echo "Запуск API сервера на порту 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Ждем немного, чтобы API успел запуститься
sleep 3

# Проверяем, что API запустился
if kill -0 $API_PID 2>/dev/null; then
    echo "✅ API сервер запущен на http://0.0.0.0:8000"
    echo "📚 API документация доступна на http://0.0.0.0:8000/docs"
else
    echo "❌ Ошибка запуска API сервера"
    exit 1
fi

# Запускаем основное приложение на 0.0.0.0 с портом 7860
echo "Запуск WebUI на порту 7860..."
python app.py --server_name 0.0.0.0 --server_port 7860 --username "$WHISPER_USERNAME" --password "$WHISPER_PASSWORD" --theme "$WHISPER_THEME" &
WEBUI_PID=$!

# Ждем завершения основного процесса
wait $WEBUI_PID
