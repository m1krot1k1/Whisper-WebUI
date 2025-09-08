# Whisper-WebUI - Установка завершена! 🎉

## Что было установлено

✅ **Системные требования:**
- Python 3.10.12
- FFmpeg 4.4.2
- NVIDIA CUDA 13.0 (RTX 4070)
- Виртуальное окружение Python

✅ **Зависимости:**
- PyTorch с поддержкой CUDA
- faster-whisper (основная реализация Whisper)
- Gradio для веб-интерфейса
- UVR для разделения музыки и вокала
- pyannote.audio для диаризации
- И множество других библиотек

✅ **Структура проекта:**
- `models/` - директории для моделей
- `outputs/` - результаты обработки
- `venv/` - виртуальное окружение
- `start-whisper-webui.sh` - скрипт запуска

## Как запустить

### Быстрый запуск:
```bash
./start-whisper-webui.sh
```

### Ручной запуск:
```bash
source venv/bin/activate
python app.py --server_name 0.0.0.0 --server_port 7860
```

## Доступ к приложению

После запуска откройте браузер и перейдите по адресу:
- **Локально:** http://localhost:7860
- **По сети:** http://YOUR_IP:7860

## Возможности

🎵 **Обработка аудио:**
- Распознавание речи (Whisper)
- Разделение музыки и вокала (UVR)
- Диаризация спикеров
- Поддержка различных форматов

🌐 **Переводы:**
- DeepL API
- Facebook NLLB модели
- Автоматическое определение языка

📝 **Форматы субтитров:**
- SRT
- WebVTT
- TXT

## Настройка моделей

Модели будут автоматически загружены при первом использовании в соответствующие директории:
- `models/Whisper/faster-whisper/` - модели faster-whisper
- `models/Whisper/insanely-fast-whisper/` - модели insanely-fast-whisper
- `models/UVR/` - модели UVR
- `models/Diarization/` - модели диаризации
- `models/NLLB/` - модели NLLB

## Дополнительные настройки

Для настройки API ключей и других параметров отредактируйте файл:
`configs/default_parameters.yaml`

## Поддержка

Если возникли проблемы:
1. Проверьте, что все зависимости установлены: `source venv/bin/activate && pip list`
2. Проверьте CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Проверьте FFmpeg: `ffmpeg -version`

## Готово к использованию! 🚀

Whisper-WebUI полностью настроен и готов к работе с вашими аудиофайлами.
