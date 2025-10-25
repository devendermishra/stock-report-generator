#!/usr/bin/env python3
"""
Test script for parallel execution of agents.
"""

import asyncio
import logging
import time
from src.main import StockReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_parallel_execution():
    """Test parallel execution of agents."""
    try:
        print("🚀 Testing Parallel Execution of Stock Report Generator")
        print("=" * 60)
        
        # Initialize the generator
        print("📋 Initializing Stock Report Generator...")
        generator = StockReportGenerator()
        
        # Test with a simple stock symbol
        stock_symbol = "TCS"
        print(f"📊 Testing with stock symbol: {stock_symbol}")
        
        # Measure execution time
        start_time = time.time()
        
        # Generate report with parallel execution
        result = await generator.generate_report(stock_symbol)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        
        if result.get("success"):
            print("✅ Report generation completed successfully!")
            print(f"📄 Report saved at: {result.get('report_path', 'N/A')}")
            
            # Show workflow results
            workflow_results = result.get("workflow_results", {})
            print(f"📈 Workflow status: {workflow_results.get('workflow_status', 'Unknown')}")
            
            # Show agent execution times if available
            if "agent_times" in workflow_results:
                print("\n🕐 Agent execution times:")
                for agent, duration in workflow_results["agent_times"].items():
                    print(f"  - {agent}: {duration:.2f}s")
        else:
            print("❌ Report generation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def test_sequential_vs_parallel():
    """Compare sequential vs parallel execution times."""
    print("\n🔄 Comparing Sequential vs Parallel Execution")
    print("=" * 60)
    
    # This would require implementing a sequential version for comparison
    # For now, just show the parallel execution
    await test_parallel_execution()

if __name__ == "__main__":
    print("🧪 Stock Report Generator - Parallel Execution Test")
    print("=" * 60)
    
    # Run the test
    asyncio.run(test_parallel_execution())
