import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file. If None, will look for .env in common locations
    """
    if env_path is None:
        # Look for .env file in common locations
        possible_paths = [
            ".env",
            "backend/configs/.env",
            "configs/.env",
            "../backend/configs/.env"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                env_path = path
                break
    
    if env_path and os.path.exists(env_path):
        print(f"Loading environment variables from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
                    print(f"Loaded: {key}")
    else:
        print("No .env file found, using system environment variables")


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variables
    
    Returns:
        HF token if found, None otherwise
    """
    # Try different possible environment variable names
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_AUTH_TOKEN")
    return token
