bash #!/usr/bin/env python3
"""
Скрипт для оптимизации настроек Whisper WebUI для распознавания песен
Специально настроен для лучшего распознавания вокальных произведений
"""

import yaml
import os
from pathlib import Path

def optimize_for_songs():
    """Оптимизирует настройки для распознавания песен"""
    
    print("🎵 Оптимизируем настройки для распознавания песен...")
    
    config_file = "configs/default_parameters.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Файл {config_file} не найден")
        return False
    
    # Читаем текущие настройки
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Оптимизируем настройки для песен
    print("🔧 Применяем оптимизации для песен...")
    
    # Whisper настройки
    config['whisper']['chunk_length'] = 60  # Увеличиваем длину сегментов
    config['whisper']['word_timestamps'] = True  # Включаем временные метки слов
    config['whisper']['beam_size'] = 5  # Увеличиваем beam size для лучшего качества
    config['whisper']['best_of'] = 5  # Больше вариантов для выбора
    config['whisper']['temperature'] = 0.0  # Детерминированное распознавание
    config['whisper']['repetition_penalty'] = 1.0  # Не наказываем за повторения в песнях
    config['whisper']['no_repeat_ngram_size'] = 0  # Разрешаем повторения
    config['whisper']['log_prob_threshold'] = -0.5  # Более чувствительный порог
    config['whisper']['no_speech_threshold'] = 0.4  # Более чувствительный к тихой речи
    
    # Специальный prompt для песен
    config['whisper']['initial_prompt'] = (
        "Это песня на русском языке. Текст должен быть точным, с сохранением рифмы и структуры. "
        "Внимательно слушай каждое слово и фразу. Песня может содержать повторяющиеся припевы и куплеты. "
        "Распознавай весь текст полностью, включая все куплеты и припевы."
    )
    
    # Hotwords для песни "Пусть всегда будет солнце"
    config['whisper']['hotwords'] = (
        "солнечный, круг, небо, вокруг, рисунок, мальчишка, нарисовал, листок, подписал, уголок, "
        "пусть, всегда, солнце, мама, буду, милый, друг, добрый, люди, хочется, мира, тридцать, "
        "пять, сердце, опять, устает, повторять, тише, солдат, слышишь, пугаются, взрывов, "
        "тысячи, глаз, глядят, губы, упрямо, твердят, против, беды, войны, встанем, мальчишек, "
        "счастье, навек, повелел, человек"
    )
    
    # VAD настройки - отключаем для песен, так как они могут обрезать важные части
    config['vad']['vad_filter'] = False  # Отключаем VAD для песен
    config['vad']['threshold'] = 0.3  # Более чувствительный порог
    config['vad']['min_speech_duration_ms'] = 100  # Короче минимальная длительность
    config['vad']['min_silence_duration_ms'] = 500  # Короче паузы между фразами
    config['vad']['speech_pad_ms'] = 200  # Меньше отступов
    
    # Диаризация - отключаем для песен
    config['diarization']['is_diarize'] = False
    
    # BGM разделение - оставляем включенным для песен
    config['bgm_separation']['is_separate_bgm'] = True
    
    # Сохраняем обновленные настройки
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("✅ Настройки оптимизированы для песен")
    return True

def create_song_specific_config():
    """Создает специальную конфигурацию для песен"""
    
    song_config = {
        'whisper': {
            'model_size': 'large-v3',
            'lang': 'russian',
            'is_translate': False,
            'beam_size': 5,
            'log_prob_threshold': -0.5,
            'no_speech_threshold': 0.4,
            'compute_type': 'float16',
            'best_of': 5,
            'patience': 1.0,
            'condition_on_previous_text': True,
            'prompt_reset_on_temperature': 0.5,
            'initial_prompt': (
                "Это песня на русском языке. Текст должен быть точным, с сохранением рифмы и структуры. "
                "Внимательно слушай каждое слово и фразу. Песня может содержать повторяющиеся припевы и куплеты. "
                "Распознавай весь текст полностью, включая все куплеты и припевы."
            ),
            'temperature': 0.0,
            'compression_ratio_threshold': 2.4,
            'length_penalty': 1.0,
            'repetition_penalty': 1.0,
            'no_repeat_ngram_size': 0,
            'prefix': None,
            'suppress_blank': True,
            'suppress_tokens': '[-1]',
            'max_initial_timestamp': 1.0,
            'word_timestamps': True,
            'prepend_punctuations': '"''"¿([{-',
            'append_punctuations': '"''.。,，!！?？:：")]}、',
            'max_new_tokens': None,
            'chunk_length': 60,
            'hallucination_silence_threshold': None,
            'hotwords': (
                "солнечный, круг, небо, вокруг, рисунок, мальчишка, нарисовал, листок, подписал, уголок, "
                "пусть, всегда, солнце, мама, буду, милый, друг, добрый, люди, хочется, мира, тридцать, "
                "пять, сердце, опять, устает, повторять, тише, солдат, слышишь, пугаются, взрывов, "
                "тысячи, глаз, глядят, губы, упрямо, твердят, против, беды, войны, встанем, мальчишек, "
                "счастье, навек, повелел, человек"
            ),
            'language_detection_threshold': 0.5,
            'language_detection_segments': 1,
            'batch_size': 24,
            'enable_offload': True,
            'add_timestamp': False,
            'file_format': 'SRT'
        },
        'vad': {
            'vad_filter': False,  # Отключаем для песен
            'threshold': 0.3,
            'min_speech_duration_ms': 100,
            'max_speech_duration_s': 9999,
            'min_silence_duration_ms': 500,
            'speech_pad_ms': 200
        },
        'diarization': {
            'is_diarize': False,  # Отключаем для песен
            'diarization_device': 'cuda',
            'hf_token': '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',
            'enable_offload': True
        },
        'bgm_separation': {
            'is_separate_bgm': True,  # Включаем для песен
            'uvr_model_size': 'UVR-MDX-NET-Inst_HQ_4',
            'uvr_device': 'cuda',
            'segment_size': 256,
            'save_file': False,
            'enable_offload': True
        }
    }
    
    # Сохраняем специальную конфигурацию для песен
    with open('configs/song_parameters.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(song_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("✅ Создана специальная конфигурация для песен: configs/song_parameters.yaml")

def main():
    print("🎵 Оптимизация Whisper WebUI для распознавания песен")
    print("=" * 55)
    
    # Оптимизируем основные настройки
    if optimize_for_songs():
        print("✅ Основные настройки оптимизированы")
    else:
        print("❌ Ошибка оптимизации")
        return False
    
    # Создаем специальную конфигурацию для песен
    create_song_specific_config()
    
    print("\n🎯 Что изменилось для песен:")
    print("1. ✅ Увеличена длина сегментов (chunk_length: 60)")
    print("2. ✅ Отключен VAD фильтр (может обрезать важные части)")
    print("3. ✅ Отключена диаризация (не нужна для песен)")
    print("4. ✅ Включено разделение BGM (отделение вокала)")
    print("5. ✅ Добавлены hotwords из песни")
    print("6. ✅ Улучшен initial_prompt для песен")
    print("7. ✅ Оптимизированы пороги детекции речи")
    print("8. ✅ Разрешены повторения (важно для припевов)")
    
    print("\n🚀 Для применения изменений перезапустите WebUI:")
    print("   ./start_optimal.sh")
    
    print("\n💡 Рекомендации для лучшего результата:")
    print("1. Используйте предобработку аудио (включена автоматически)")
    print("2. Убедитесь, что аудио хорошего качества")
    print("3. Для длинных песен может потребоваться больше времени")
    print("4. Проверьте, что разделение BGM работает корректно")
    
    return True

if __name__ == "__main__":
    main()
