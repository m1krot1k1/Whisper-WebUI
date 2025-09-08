#!/bin/bash

# Whisper-WebUI Startup Script
# This script starts the Whisper-WebUI application

echo "Starting Whisper-WebUI..."
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run Install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    echo "CUDA check passed."
else
    echo "Warning: CUDA check failed, but continuing..."
fi

echo ""
echo "Starting Whisper-WebUI on http://0.0.0.0:7860"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python app.py --server_name 0.0.0.0 --server_port 7860 --inbrowser false
