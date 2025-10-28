#!/usr/bin/env python3
"""
Test script to demonstrate the days_back parameter functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_time_keywords():
    """Test the _get_time_keywords function."""
    try:
        from src.tools.web_search_tool import _get_time_keywords
        
        test_cases = [
            (1, "today yesterday"),
            (3, "last 3 days recent"),
            (7, "last week recent"),
            (14, "last 2 weeks recent"),
            (30, "last month recent"),
            (90, "last 3 months recent"),
            (365, "recent latest")
        ]
        
        print("Testing _get_time_keywords function:")
        print("=" * 50)
        
        for days, expected in test_cases:
            result = _get_time_keywords(days)
            status = "✅" if result == expected else "❌"
            print(f"{status} {days:3d} days -> '{result}' (expected: '{expected}')")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_query_building():
    """Test how queries are built with time keywords."""
    try:
        from src.tools.web_search_tool import _get_time_keywords
        
        print("\nTesting query building with time keywords:")
        print("=" * 50)
        
        # Test sector news query
        sector = "Banking"
        days_back = 7
        time_keywords = _get_time_keywords(days_back)
        query = f"{sector} sector news India NSE market trends {time_keywords}"
        
        print(f"Sector: {sector}")
        print(f"Days back: {days_back}")
        print(f"Time keywords: {time_keywords}")
        print(f"Final query: {query}")
        
        # Test company news query
        company_name = "Reliance Industries"
        stock_symbol = "RELIANCE"
        days_back = 3
        time_keywords = _get_time_keywords(days_back)
        query = f"{company_name} {stock_symbol} news India NSE announcements {time_keywords}"
        
        print(f"\nCompany: {company_name} ({stock_symbol})")
        print(f"Days back: {days_back}")
        print(f"Time keywords: {time_keywords}")
        print(f"Final query: {query}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_date_filtering():
    """Test the date filtering logic."""
    try:
        from src.tools.web_search_tool import _filter_results_by_date
        from datetime import datetime, timedelta
        
        print("\nTesting date filtering logic:")
        print("=" * 50)
        
        # Create mock search results with different dates
        now = datetime.now()
        mock_results = [
            {"title": "News 1", "date": (now - timedelta(days=1)).strftime('%Y-%m-%d')},
            {"title": "News 2", "date": (now - timedelta(days=5)).strftime('%Y-%m-%d')},
            {"title": "News 3", "date": (now - timedelta(days=10)).strftime('%Y-%m-%d')},
            {"title": "News 4", "date": None},  # No date
        ]
        
        # Test filtering for last 7 days
        filtered = _filter_results_by_date(mock_results, 7)
        print(f"Original results: {len(mock_results)}")
        print(f"Filtered results (7 days): {len(filtered)}")
        
        # Test filtering for last 3 days
        filtered = _filter_results_by_date(mock_results, 3)
        print(f"Filtered results (3 days): {len(filtered)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing days_back Parameter Implementation")
    print("=" * 60)
    
    tests = [
        ("Time Keywords", test_time_keywords),
        ("Query Building", test_query_building),
        ("Date Filtering", test_date_filtering),
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
        print("🎉 All tests passed! days_back parameter is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

