#!/bin/bash

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected. Installing cuDNN 8 for WhisperX compatibility..."
        sudo apt update
        sudo apt install -y libcudnn8-dev
        echo "cuDNN 8 installed successfully."
    else
        echo "CUDA not detected. WhisperX will use CPU mode."
    fi
}

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

python -m pip install -U pip

# Check and install CUDA dependencies
check_cuda

# Install Python requirements
pip install -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    deactivate
    exit 1
}

echo ""
echo "Installation completed successfully!"
echo "You can now run the application with: bash start-webui.sh"
echo ""
echo "Available Whisper implementations:"
echo "  - faster-whisper (default)"
echo "  - whisper (original OpenAI)"
echo "  - whisperx (enhanced quality)"
echo ""
echo "To change implementation, edit backend/configs/config.yaml:"
echo "  implementation: whisperx  # or faster-whisper or whisper"

deactivate
