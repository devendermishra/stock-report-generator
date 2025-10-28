#!/usr/bin/env python3
"""
Test script to verify LangChain tools are working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tool_imports():
    """Test that all tools can be imported correctly."""
    try:
        from src.tools.langchain_tools import get_all_tools, initialize_tools, TOOL_DESCRIPTIONS
        print("✅ Successfully imported langchain_tools")
        
        # Test tool initialization
        tools = get_all_tools()
        print(f"✅ Found {len(tools)} LangChain tools")
        
        # Print tool names
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_individual_tools():
    """Test individual tool imports."""
    try:
        from src.tools.stock_data_tool import get_stock_metrics, get_company_info, validate_symbol
        print("✅ Successfully imported stock data tools")
        
        from src.tools.web_search_tool import search_sector_news, search_company_news, search_market_trends
        print("✅ Successfully imported web search tools")
        
        from src.tools.summarizer_tool import summarize_text, extract_insights
        print("✅ Successfully imported summarizer tools")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tool_metadata():
    """Test that tools have proper metadata."""
    try:
        from src.tools.stock_data_tool import get_stock_metrics
        
        # Check that the tool has the required attributes
        assert hasattr(get_stock_metrics, 'name'), "Tool missing 'name' attribute"
        assert hasattr(get_stock_metrics, 'description'), "Tool missing 'description' attribute"
        assert hasattr(get_stock_metrics, 'args_schema'), "Tool missing 'args_schema' attribute"
        
        print(f"✅ Tool metadata check passed for {get_stock_metrics.name}")
        return True
        
    except Exception as e:
        print(f"❌ Tool metadata error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing LangChain Tools Implementation")
    print("=" * 50)
    
    tests = [
        ("Tool Imports", test_tool_imports),
        ("Individual Tools", test_individual_tools),
        ("Tool Metadata", test_tool_metadata),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangChain tools are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

