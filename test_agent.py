#!/usr/bin/env python3
"""Simple test script to check if the graph agent works."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic agent functionality."""
    print("🧪 Testing basic agent functionality...")
    
    try:
        # Import the main function
        from graph_agent import execute_banking_graph
        print("✅ Successfully imported execute_banking_graph")
        
        # Test with a simple greeting
        print("\n🧪 Testing greeting...")
        result = execute_banking_graph(
            thread_id="test_thread_001",
            user_input="hi",
            user_token="test_user",
            current_state=None
        )
        
        print(f"✅ Result: {result}")
        
        if result and "message" in result:
            print(f"✅ Got response: {result['message']}")
            return True
        else:
            print("❌ No proper response received")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transaction_intent():
    """Test transaction intent detection."""
    print("\n🧪 Testing transaction intent...")
    
    try:
        from graph_agent import execute_banking_graph
        
        result = execute_banking_graph(
            thread_id="test_thread_002",
            user_input="can you do a transaction?",
            user_token="test_user",
            current_state=None
        )
        
        print(f"✅ Transaction test result: {result}")
        
        if result and "message" in result:
            print(f"✅ Got response: {result['message']}")
            return True
        else:
            print("❌ No proper response received")
            return False
            
    except Exception as e:
        print(f"❌ Error during transaction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Graph Agent Tests\n")
    
    # Run tests
    test1_passed = test_basic_functionality()
    test2_passed = test_transaction_intent()
    
    print(f"\n📋 Test Results:")
    print(f"   Greeting test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Transaction test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The agent is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
