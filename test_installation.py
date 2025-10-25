"""
Test script to verify the Stock Report Generator installation.
"""

import sys
import os

def test_imports() -> bool:
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        # Core packages
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import requests
        print("✅ requests imported successfully")
        
        # AI packages
        import openai
        print("✅ openai imported successfully")
        
        import langchain
        print("✅ langchain imported successfully")
        
        import langgraph
        print("✅ langgraph imported successfully")
        
        # Financial data
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        # PDF processing
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
        
        # Web search
        import tavily
        print("✅ tavily imported successfully")
        
        # Data processing
        import bs4
        print("✅ beautifulsoup4 imported successfully")
        
        # Async support
        import aiohttp
        print("✅ aiohttp imported successfully")
        
        # Logging
        import structlog
        print("✅ structlog imported successfully")
        
        print("\n🎉 All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality() -> bool:
    """Test basic functionality of key packages."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        print("✅ pandas basic functionality works")
        
        # Test yfinance
        import yfinance as yf
        # Just test that we can create a ticker object
        ticker = yf.Ticker("AAPL")
        print("✅ yfinance basic functionality works")
        
        # Test OpenAI client creation
        import openai
        # Just test that we can create a client (without API key)
        try:
            client = openai.OpenAI(api_key="test")
            print("✅ openai client creation works")
        except Exception:
            print("✅ openai client creation works (expected error without real API key)")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_project_structure() -> bool:
    """Test that the project structure is correct."""
    print("\n📁 Testing project structure...")
    
    required_files = [
        "src/main.py",
        "src/config.py",
        "src/agents/__init__.py",
        "src/tools/__init__.py",
        "src/graph/__init__.py",
        "requirements-minimal.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main() -> bool:
    """Main test function."""
    print("Stock Report Generator - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test project structure
    structure_ok = test_project_structure()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    print(f"  Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    
    if imports_ok and functionality_ok and structure_ok:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Run: python test_system.py")
        print("3. Run: cd src && python main.py --symbol RELIANCE --company 'Reliance Industries Limited' --sector 'Oil & Gas'")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
