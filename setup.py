"""
Setup script for Stock Report Generator.
Handles installation and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies() -> bool:
    """Install required dependencies."""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories() -> bool:
    """Create necessary directories."""
    directories = ["reports", "temp", "logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def setup_environment() -> bool:
    """Set up environment configuration."""
    env_example = "env.example"
    env_file = ".env"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print(f"📄 Created .env file from {env_example}")
            print("⚠️  Please edit .env file with your API keys")
        else:
            print("⚠️  No env.example file found")
    else:
        print("✅ .env file already exists")
    
    return True

def verify_setup() -> bool:
    """Verify the setup is correct."""
    print("\n🔍 Verifying setup...")
    
    # Check if src directory exists
    if not os.path.exists("src"):
        print("❌ src directory not found")
        return False
    
    # Check if main.py exists
    if not os.path.exists("src/main.py"):
        print("❌ src/main.py not found")
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    print("✅ Setup verification passed")
    return True

def main() -> bool:
    """Main setup function."""
    print("Stock Report Generator - Setup")
    print("=============================")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Verify setup
    if not verify_setup():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python test_system.py")
    print("3. Run: cd src && python main.py --symbol RELIANCE --company 'Reliance Industries Limited' --sector 'Oil & Gas'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
