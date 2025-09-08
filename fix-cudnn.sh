#!/bin/bash

echo "🔧 Whisper-WebUI cuDNN Fix Script"
echo "================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script needs to be run as root to fix cuDNN issues."
    echo "   Please run: sudo bash fix-cudnn.sh"
    exit 1
fi

# Function to create cuDNN compatibility symlinks
create_cudnn_symlinks() {
    echo "Creating cuDNN compatibility symlinks..."
    cd /usr/lib/x86_64-linux-gnu
    for lib in libcudnn.so libcudnn_adv.so libcudnn_ops.so libcudnn_heuristic.so libcudnn_engines_precompiled.so libcudnn_engines_runtime_compiled.so libcudnn_graph.so libcudnn_cnn.so; do
        if [ -f "${lib}.9.13.0" ]; then
            ln -sf ${lib}.9.13.0 ${lib}.9.1.0 2>/dev/null || true
            ln -sf ${lib}.9.13.0 ${lib}.9.1 2>/dev/null || true
            ln -sf ${lib}.9.13.0 ${lib}.9 2>/dev/null || true
            echo "   ✅ Created symlinks for ${lib}"
        elif [ -f "${lib}.8.9.7" ]; then
            ln -sf ${lib}.8.9.7 ${lib}.9.1.0 2>/dev/null || true
            ln -sf ${lib}.8.9.7 ${lib}.9.1 2>/dev/null || true
            ln -sf ${lib}.8.9.7 ${lib}.9 2>/dev/null || true
            echo "   ✅ Created symlinks for ${lib} (from 8.9.7)"
        fi
    done
    cd - > /dev/null
    ldconfig
    echo "Compatibility symlinks created."
}

# Check if cuDNN libraries exist
if ! find /usr -name "libcudnn_cnn.so*" 2>/dev/null | grep -q .; then
    echo "❌ cuDNN libraries not found!"
    echo "   Please run: sudo bash Install.sh"
    exit 1
fi

echo "✅ cuDNN libraries found"

# Check PyTorch cuDNN version
echo "Checking PyTorch cuDNN version..."
cd /root/Whisper-WebUI
if [ -d "venv" ]; then
    source venv/bin/activate
    PYTORCH_CUDNN=$(python -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null)
    echo "PyTorch cuDNN version: $PYTORCH_CUDNN"
    
    if [ "$PYTORCH_CUDNN" != "90100" ]; then
        echo "⚠️  PyTorch cuDNN version mismatch detected!"
        echo "   Reinstalling PyTorch with compatible cuDNN..."
        pip uninstall -y torch torchvision torchaudio
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo "✅ PyTorch reinstalled with compatible cuDNN"
    else
        echo "✅ PyTorch cuDNN version is correct"
    fi
    deactivate
else
    echo "⚠️  Virtual environment not found. Please run: bash Install.sh"
fi

create_cudnn_symlinks

echo ""
echo "🎉 cuDNN fix completed!"
echo "   You can now run: bash start-webui.sh"
