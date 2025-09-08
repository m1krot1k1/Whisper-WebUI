#!/bin/bash

# Function to create cuDNN compatibility symlinks
create_cudnn_symlinks() {
    echo "Creating cuDNN compatibility symlinks..."
    cd /usr/lib/x86_64-linux-gnu
    
    # Create symlinks for all cuDNN libraries
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

# Function to install cuDNN libraries
install_cudnn() {
    echo "Checking cuDNN installation..."
    
    # Check if cuDNN libraries are already installed
    if find /usr -name "libcudnn_cnn.so*" 2>/dev/null | grep -q .; then
        echo "cuDNN libraries are already installed."
        # Create compatibility symlinks even if cuDNN is already installed
        create_cudnn_symlinks
        return 0
    fi
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo ""
        echo "⚠️  cuDNN libraries are not installed and you're not running as root."
        echo "   To install cuDNN libraries, run: sudo bash $0"
        echo "   Or install them manually:"
        echo ""
        echo "   # Add NVIDIA CUDA repository"
        echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        echo "   sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        echo "   rm cuda-keyring_1.1-1_all.deb"
        echo ""
        echo "   # Update package list"
        echo "   sudo apt update"
        echo ""
        echo "   # Install cuDNN (replace with your CUDA version)"
        echo "   sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13"
        echo ""
        echo "   # Update library cache"
        echo "   sudo ldconfig"
        echo ""
        echo "   Then run this script again."
        echo ""
        echo "Continuing with Python environment setup..."
        return 1
    fi
    
    echo "Installing cuDNN libraries..."
    
    # Add NVIDIA CUDA repository if not already added
    if ! apt list --installed | grep -q cuda-keyring; then
        echo "Adding NVIDIA CUDA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
    fi
    
    # Update package list
    echo "Updating package list..."
    apt update -qq
    
    # Detect CUDA version and install appropriate cuDNN
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    echo "Detected CUDA version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "13."* ]]; then
        echo "Installing cuDNN for CUDA 13.x..."
        apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing cuDNN for CUDA 12.x..."
        apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "Installing cuDNN for CUDA 11.x..."
        apt install -y libcudnn9-cuda-11 libcudnn9-dev-cuda-11 libcudnn9-headers-cuda-11
    else
        echo "Warning: Unsupported CUDA version $CUDA_VERSION. Attempting to install cuDNN for CUDA 12.x..."
        apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
    fi
    
    # Update library cache
    ldconfig
    
    # Create compatibility symlinks for older cuDNN versions
    create_cudnn_symlinks
    
    echo "cuDNN installation completed."
    return 0
}

# Install cuDNN libraries (if possible)
install_cudnn

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

python -m pip install -U pip

# Install PyTorch with compatible cuDNN version first
echo "Installing PyTorch with compatible cuDNN..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    deactivate
    exit 1
}

deactivate

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Summary:"
echo "   ✅ Python virtual environment created"
echo "   ✅ Python dependencies installed"
if find /usr -name "libcudnn_cnn.so*" 2>/dev/null | grep -q .; then
    echo "   ✅ cuDNN libraries available"
else
    echo "   ⚠️  cuDNN libraries not installed (may cause errors)"
fi
echo ""
echo "🚀 To run the application:"
echo "   bash start-webui.sh"
echo ""
echo "🌐 The application will be available at: http://127.0.0.1:7860"
