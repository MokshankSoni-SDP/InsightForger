"""
Setup script for Autonomous Insight Engine.

Run this script to set up the environment and verify installation.
"""
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    dirs = [
        "temp_uploads",
        "reports"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    return True


def check_env_file():
    """Check if .env file exists and has keys."""
    print("\nChecking .env file...")
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print("  ⚠️ .env file not found")
        print("  Creating template .env file...")
        
        with open(env_path, "w") as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
            f.write("HF_API_KEY=your_huggingface_api_key_here\n")
            f.write("GROQ_MODEL=groq/llama-3.1-70b-versatile\n")
            f.write("HF_MODEL=huggingface/meta-llama/Meta-Llama-3-8B-Instruct\n")
            f.write("MAX_RETRIES=3\n")
            f.write("USE_GPU=true\n")
        
        print("  ✓ .env file created")
        print("\n  ⚠️ IMPORTANT: Edit .env and add your API keys!")
        return False
    
    # Check if keys are set
    with open(env_path) as f:
        content = f.read()
    
    if "your_groq_api_key_here" in content or "your_huggingface_api_key_here" in content:
        print("  ⚠️ API keys not configured")
        print("  Please edit .env and add your actual API keys")
        return False
    
    print("  ✓ .env file configured")
    return True


def check_gpu():
    """Check if GPU is available."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ℹ️ No GPU detected - will use CPU")
            return True
    except ImportError:
        print("  ℹ️ PyTorch not installed yet - GPU check will happen after installation")
        return True


def main():
    """Run setup."""
    print("=" * 60)
    print("AUTONOMOUS INSIGHT ENGINE - SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check .env
    env_configured = check_env_file()
    
    # Check GPU
    check_gpu()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    
    if not env_configured:
        print("\n⚠️ NEXT STEPS:")
        print("1. Edit .env and add your API keys")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run: streamlit run app.py")
    else:
        print("\n✅ READY TO USE")
        print("\nTo start the application:")
        print("  streamlit run app.py")
        print("\nOr run via command line:")
        print("  python main.py your_data.csv")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
