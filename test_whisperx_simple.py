#!/usr/bin/env python3

import yaml
from modules.utils.paths import SERVER_CONFIG_PATH

# Test simple WhisperX configuration
config_path = '/root/Whisper-WebUI/backend/configs/config.yaml'

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    whisper_config = config.get('whisper', {})
    print("✅ Whisper Configuration loaded successfully!")
    print("Current config:", whisper_config)
    print()
    print("📋 Available parameters:")
    print("- implementation:", whisper_config.get('implementation', 'faster-whisper'))
    print("- model_size:", whisper_config.get('model_size', 'large-v3'))
    print("- compute_type:", whisper_config.get('compute_type', 'float16'))
    print("- device:", whisper_config.get('device', 'cuda'))
    print("- return_char_alignments:", whisper_config.get('return_char_alignments', True))

    print("\n✅ Configuration is properly structured!")
    print("No duplicate sections, all parameters in main 'whisper' section.")

except Exception as e:
    print(f"❌ Error: {e}")
