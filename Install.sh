#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

# Allow continuing after non-critical failures in specific steps
trap 'echo "[!] Script failed at line $LINENO"' ERR

FORCE_CPU=${FORCE_CPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
INSTALLED_TORCH_TAG=""

# Colored output helpers
bold() { echo -e "\e[1m$*\e[0m"; }
green() { echo -e "\e[32m$*\e[0m"; }
yellow() { echo -e "\e[33m$*\e[0m"; }
red() { echo -e "\e[31m$*\e[0m"; }

# Install system dependencies
echo "Installing system dependencies..."
if [ "$EUID" -eq 0 ]; then
    apt update -qq && apt install -y ffmpeg
else
    sudo apt update -qq && sudo apt install -y ffmpeg
fi
echo "System dependencies installed."

# Function to create cuDNN compatibility symlinks
create_cudnn_symlinks() {
    echo "Creating cuDNN compatibility symlinks..."
    local cudnn_dir="/usr/lib/x86_64-linux-gnu"
    if [ ! -d "$cudnn_dir" ]; then
        yellow "   cuDNN directory not found: $cudnn_dir"
        return 0
    fi
    pushd "$cudnn_dir" >/dev/null || return 0
    # Libraries we care about
    local libs=(libcudnn libcudnn_adv libcudnn_ops libcudnn_heuristic libcudnn_engines_precompiled libcudnn_engines_runtime_compiled libcudnn_graph libcudnn_cnn)
    for base in "${libs[@]}"; do
        # Find the highest version available (prefer 9.*, fallback to 8.*)
        target=$(ls -1 ${base}.so.* 2>/dev/null | sort -V | grep -E '\.9\.' || true | tail -n1)
        if [ -z "$target" ]; then
            target=$(ls -1 ${base}.so.* 2>/dev/null | sort -V | tail -n1 || true)
        fi
        if [ -n "$target" ]; then
            for linkver in 9.1.0 9.1 9; do
                ln -sf "$target" "${base}.so.${linkver}" 2>/dev/null || true
            done
            echo "   ✅ ${base} -> $target (compat links)"
        fi
    done
    popd >/dev/null || true
    ldconfig || true
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

# Install cuDNN libraries (if possible & not forced CPU)
if [ "$FORCE_CPU" != "1" ]; then
  install_cudnn || true
else
  yellow "Skipping cuDNN install (FORCE_CPU=1)"
fi


# Check if python3 is installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing..."
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

python -m pip install -U pip wheel setuptools

PYTORCH_TAG="cpu"
if [ "$FORCE_CPU" != "1" ] && command -v nvidia-smi &>/dev/null; then
  DETECTED_CUDA=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/' || echo "0.0")
  case "$DETECTED_CUDA" in
    13.*) PYTORCH_TAG="cu130" ;; # may not exist yet -> fallback logic below
    12.*) PYTORCH_TAG="cu121" ;;
    11.*) PYTORCH_TAG="cu118" ;;
    *) PYTORCH_TAG="cpu" ;;
  esac
fi

echo "Requested PyTorch CUDA tag: $PYTORCH_TAG"

install_torch() {
  local tag="$1"
  if [ "$tag" = "cpu" ]; then
     pip install --no-cache-dir torch torchvision torchaudio && return 0
  else
     pip install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${tag}" && return 0
  fi
  return 1
}

# Attempt install with fallback sequence
FALLBACKS=()
if [ "$PYTORCH_TAG" = "cu130" ]; then
  FALLBACKS=(cu130 cu121 cu118 cpu)
elif [ "$PYTORCH_TAG" = "cu121" ]; then
  FALLBACKS=(cu121 cu118 cpu)
elif [ "$PYTORCH_TAG" = "cu118" ]; then
  FALLBACKS=(cu118 cu121 cpu)
else
  FALLBACKS=(cpu)
fi

INSTALLED_TORCH_TAG=""
for tag in "${FALLBACKS[@]}"; do
  echo "Trying PyTorch build: $tag"
  if install_torch "$tag"; then
     INSTALLED_TORCH_TAG="$tag"
     green "Installed PyTorch tag: $tag"
     break
  else
     yellow "Failed to install tag $tag, trying next..."
  fi
done

if [ -z "$INSTALLED_TORCH_TAG" ]; then
  red "Failed to install any PyTorch build. Aborting."
  deactivate || true
  exit 1
fi

# Quick torch sanity test
python - <<'PY'
import torch, sys
print('Torch version:', torch.__version__)
print('Built with CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
print('cuDNN available:', torch.backends.cudnn.is_available())
print('cuDNN version:', torch.backends.cudnn.version())
PY

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
echo "   ✅ PyTorch tag used: ${INSTALLED_TORCH_TAG}"
if [ "$FORCE_CPU" = "1" ]; then
  echo "   ℹ️  Forced CPU mode (FORCE_CPU=1)"
fi
echo ""
echo "🚀 To run the application:"
echo "   bash start-webui.sh"
echo ""
echo "🌐 The application will be available at: http://127.0.0.1:7860"
