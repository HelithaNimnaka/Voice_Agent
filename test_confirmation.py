#!/usr/bin/env python3
"""Test script for confirmation and transaction completion flow."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_confirmation_and_completion():
    """Test the complete flow from start to transaction completion."""
    print("🧪 Testing confirmation and completion flow...")
    
    try:
        from graph_agent import execute_banking_graph
        
        # Step 1: Set up complete transaction
        print("\n1. Setting up complete transaction...")
        result1 = execute_banking_graph(
            thread_id="test_thread_confirm",
            user_input="transfer 100 to Alice",
            user_token="test_user",
            current_state=None
        )
        
        print(f"Step 1 - Response: {result1.get('message', 'No response')}")
        
        # Verify we got confirmation request
        if "confirm" not in result1.get('message', '').lower():
            print("❌ Expected confirmation request but didn't get one")
            return False
        
        # Step 2: Confirm the transaction
        print("\n2. Confirming transaction...")
        result2 = execute_banking_graph(
            thread_id="test_thread_confirm",
            user_input="yes",
            user_token="test_user",
            current_state=result1.get('transfer_state')
        )
        
        transfer_state = result2.get('transfer_state', {})
        transaction_result = transfer_state.get('transaction_result', '')
        
        print(f"Step 2 - Response: {result2.get('message', 'No response')}")
        print(f"Step 2 - Transaction Result: {transaction_result}")
        
        # Check if transaction was completed successfully
        if transaction_result == "COMPLETED":
            print("✅ Transaction completed successfully!")
            return True
        elif transaction_result == "ERROR":
            print("❌ Transaction failed with error")
            return False
        else:
            print(f"❓ Unexpected transaction result: {transaction_result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during confirmation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decline_transaction():
    """Test declining a transaction."""
    print("\n🧪 Testing transaction decline...")
    
    try:
        from graph_agent import execute_banking_graph
        
        # Step 1: Set up transaction
        print("\n1. Setting up transaction...")
        result1 = execute_banking_graph(
            thread_id="test_thread_decline",
            user_input="transfer 50 to Bob",
            user_token="test_user",
            current_state=None
        )
        
        # Step 2: Decline the transaction
        print("\n2. Declining transaction...")
        result2 = execute_banking_graph(
            thread_id="test_thread_decline",
            user_input="no",
            user_token="test_user",
            current_state=result1.get('transfer_state')
        )
        
        transfer_state = result2.get('transfer_state', {})
        transaction_result = transfer_state.get('transaction_result', '')
        
        print(f"Step 2 - Response: {result2.get('message', 'No response')}")
        print(f"Step 2 - Transaction Result: {transaction_result}")
        
        # Check if transaction was cancelled
        if transaction_result == "CANCELLED":
            print("✅ Transaction cancelled successfully!")
            return True
        else:
            print(f"❌ Expected CANCELLED but got: {transaction_result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during decline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all confirmation tests."""
    print("🚀 Testing Confirmation Flow\n")
    
    # Test confirmation and completion
    confirm_result = test_confirmation_and_completion()
    
    # Test declining transaction
    decline_result = test_decline_transaction()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   Confirmation & Completion: {'✅ PASSED' if confirm_result else '❌ FAILED'}")
    print(f"   Transaction Decline: {'✅ PASSED' if decline_result else '❌ FAILED'}")
    
    if confirm_result and decline_result:
        print("\n🎉 All confirmation tests passed!")
    else:
        print("\n⚠️ Some confirmation tests failed.")

if __name__ == "__main__":
    main()
