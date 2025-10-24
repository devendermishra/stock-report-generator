"""
Test script for Stock Report Generator system.
Demonstrates the system capabilities with a simple example.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append('src')

from src.main import StockReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_system():
    """Test the Stock Report Generator system."""
    try:
        print("🚀 Testing Stock Report Generator System")
        print("=" * 50)
        
        # Initialize the generator
        print("📋 Initializing system...")
        generator = StockReportGenerator()
        
        if not generator.initialize():
            print("❌ Failed to initialize system")
            return False
            
        print("✅ System initialized successfully")
        
        # Test with a sample stock
        test_cases = [
            {
                "symbol": "RELIANCE",
                "company": "Reliance Industries Limited",
                "sector": "Oil & Gas"
            },
            {
                "symbol": "TCS", 
                "company": "Tata Consultancy Services",
                "sector": "IT"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 Test Case {i}: {test_case['symbol']}")
            print("-" * 30)
            
            # Generate report
            result = await generator.generate_report(
                stock_symbol=test_case["symbol"],
                company_name=test_case["company"],
                sector=test_case["sector"]
            )
            
            if result["success"]:
                print(f"✅ Report generated successfully for {test_case['symbol']}")
                print(f"📄 Report saved to: {result.get('report_path', 'N/A')}")
                print(f"⏱️  Generated at: {result.get('generated_at', 'N/A')}")
                
                # Check workflow status
                status = generator.get_report_status(test_case["symbol"])
                if status["success"]:
                    print(f"📈 Workflow status: {status['status']['workflow_status']}")
                    
            else:
                print(f"❌ Report generation failed for {test_case['symbol']}")
                print(f"🔍 Error: {result.get('error', 'Unknown error')}")
                
        print("\n🎉 System test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Stock Report Generator - System Test")
    print("====================================")
    
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
        
    # Run the test
    success = asyncio.run(test_system())
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    return success

if __name__ == "__main__":
    main()
