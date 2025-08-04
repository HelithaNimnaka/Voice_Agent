#!/usr/bin/env python3
"""Test script for amount extraction and transaction flow."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_amount_extraction():
    """Test amount extraction in transaction context."""
    print("🧪 Testing amount extraction...")
    
    try:
        from graph_agent import execute_banking_graph
        
        # First, set up a transaction context with a payee
        print("\n1. Setting up transaction with payee...")
        result1 = execute_banking_graph(
            thread_id="test_thread_amount",
            user_input="transfer to Alice",
            user_token="test_user",
            current_state=None
        )
        
        print(f"Step 1 - Transfer state: {result1.get('transfer_state', {}).get('destination_account', 'None')}")
        
        # Now test providing just an amount
        print("\n2. Providing amount...")
        result2 = execute_banking_graph(
            thread_id="test_thread_amount", 
            user_input="100",
            user_token="test_user",
            current_state=result1.get('transfer_state')
        )
        
        transfer_state = result2.get('transfer_state', {})
        destination = transfer_state.get('destination_account', '')
        amount = transfer_state.get('transfer_amount', '')
        
        print(f"Step 2 - Destination: '{destination}', Amount: '{amount}'")
        print(f"Step 2 - Response: {result2.get('message', 'No response')}")
        
        # Check if we got confirmation request
        if "confirm" in result2.get('message', '').lower():
            print("✅ Amount extraction successful - got confirmation request")
            return True
        else:
            print("❌ Amount extraction failed - no confirmation request")
            return False
            
    except Exception as e:
        print(f"❌ Error during amount test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transaction_completion():
    """Test complete transaction flow."""
    print("\n🧪 Testing complete transaction flow...")
    
    try:
        from graph_agent import execute_banking_graph
        
        # Full transaction in one go
        print("\n1. Complete transaction details...")
        result = execute_banking_graph(
            thread_id="test_thread_complete",
            user_input="transfer 100 to Alice",
            user_token="test_user",
            current_state=None
        )
        
        transfer_state = result.get('transfer_state', {})
        destination = transfer_state.get('destination_account', '')
        amount = transfer_state.get('transfer_amount', '')
        
        print(f"Complete - Destination: '{destination}', Amount: '{amount}'")
        print(f"Complete - Response: {result.get('message', 'No response')}")
        
        # Should ask for confirmation
        if "confirm" in result.get('message', '').lower() and destination and amount:
            print("✅ Complete transaction setup successful")
            return True
        else:
            print("❌ Complete transaction setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during complete transaction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Transaction Flow\n")
    
    test1_passed = test_amount_extraction()
    test2_passed = test_transaction_completion()
    
    print(f"\n📋 Test Results:")
    print(f"   Amount extraction: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Complete transaction: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 Transaction flow tests passed!")
    else:
        print("\n⚠️ Some transaction tests failed.")
