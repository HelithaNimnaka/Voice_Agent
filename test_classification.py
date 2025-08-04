#!/usr/bin/env python3
"""
Enhanced Classification Testing Suite
Tests the improved LLM classification system for banking agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_agent import (
    analyze_user_intent_and_extract_data, 
    handle_general_inquiry, 
    classify_user_confirmation,
    TransferState
)

def test_intent_classification():
    """Test intent classification for various user inputs"""
    print("🧪 TESTING INTENT CLASSIFICATION")
    print("=" * 60)
    
    # Test cases from conversation logs and edge cases
    test_cases = [
        # Original problematic cases
        ("thanks buddy", "general", "Should classify as general/courtesy"),
        ("shall we do a transaction", "transaction", "Should recognize transaction intent"),
        ("let's do a transaction", "transaction", "Should recognize casual transaction language"),
        ("that's all", "general", "Should classify as general/courtesy"),
        ("cancel that", "general", "Should classify as general/cancellation"),
        
        # Transaction variations
        ("move money", "transaction", "Should recognize money movement intent"),
        ("do a transfer", "transaction", "Should recognize transfer intent"),
        ("start a transaction", "transaction", "Should recognize transaction start"),
        ("send money to John", "transaction", "Should recognize send money intent"),
        ("transfer 500", "transaction", "Should recognize amount-based transaction"),
        
        # Cancellation variations
        ("cancel", "general", "Should classify as cancellation"),
        ("stop", "general", "Should classify as cancellation"),
        ("never mind", "general", "Should classify as cancellation"),
        ("abort", "general", "Should classify as cancellation"),
        
        # Courtesy variations
        ("all good", "general", "Should classify as courtesy"),
        ("we're done", "general", "Should classify as courtesy"),
        ("thank you", "general", "Should classify as courtesy"),
        ("appreciate it", "general", "Should classify as courtesy"),
        
        # Banking functions
        ("check my balance", "balance_inquiry", "Should recognize balance inquiry"),
        ("what is my balance", "balance_inquiry", "Should recognize balance question"),
        ("show my payees", "payee_list", "Should recognize payee list request"),
        ("who are my contacts", "payee_list", "Should recognize contacts request"),
        ("add new payee", "add_payee", "Should recognize add payee request"),
        
        # Compound requests
        ("show payees and balance", "compound", "Should recognize compound request"),
        ("balance and payee list", "compound", "Should recognize compound request"),
        
        # Greetings
        ("hello", "general", "Should classify as greeting"),
        ("hi there", "general", "Should classify as greeting"),
        ("good morning", "general", "Should classify as greeting"),
    ]
    
    state = TransferState()
    state['user_token'] = 'test_user'
    state['chat_history'] = []
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (test_input, expected_intent, description) in enumerate(test_cases, 1):
        print(f"\nTest {i:2d}: '{test_input}'")
        print(f"        Expected: {expected_intent}")
        print(f"        Description: {description}")
        
        try:
            result = analyze_user_intent_and_extract_data(test_input, 'test_user', state)
            actual_intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0)
            account = result.get('account', 'None')
            amount = result.get('amount', None)
            
            if actual_intent == expected_intent:
                print(f"        ✅ PASS: {actual_intent} (confidence: {confidence}%)")
                if account != 'None':
                    print(f"           Account: {account}")
                if amount is not None:
                    print(f"           Amount: {amount}")
                success_count += 1
            else:
                print(f"        ❌ FAIL: Got {actual_intent}, expected {expected_intent}")
                print(f"           Confidence: {confidence}%")
                
        except Exception as e:
            print(f"        ❌ ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Intent Classification Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    return success_count, total_tests

def test_general_inquiry_handling():
    """Test general inquiry handling and classification"""
    print("\n🧪 TESTING GENERAL INQUIRY HANDLING")
    print("=" * 60)
    
    test_cases = [
        ("thanks buddy", "courtesy", "Should handle buddy thanks"),
        ("thank you so much", "courtesy", "Should handle appreciation"),
        ("that's all", "courtesy", "Should handle completion phrase"),
        ("all good", "courtesy", "Should handle positive completion"),
        ("we're done", "courtesy", "Should handle conversation end"),
        ("cancel that", "cancellation", "Should handle cancellation request"),
        ("cancel", "cancellation", "Should handle simple cancel"),
        ("stop", "cancellation", "Should handle stop command"),
        ("hello", "greeting", "Should handle greeting"),
        ("hi there", "greeting", "Should handle casual greeting"),
        ("good morning", "greeting", "Should handle formal greeting"),
        ("how are you", "other", "Should handle non-banking query"),
    ]
    
    state = TransferState()
    state['user_token'] = 'test_user'
    state['chat_history'] = []
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (test_input, expected_type, description) in enumerate(test_cases, 1):
        print(f"\nTest {i:2d}: '{test_input}'")
        print(f"        Expected type: {expected_type}")
        print(f"        Description: {description}")
        
        try:
            response = handle_general_inquiry(test_input, 'test_user', state)
            print(f"        ✅ Response: {response[:80]}...")
            success_count += 1
            
        except Exception as e:
            print(f"        ❌ ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"General Inquiry Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    return success_count, total_tests

def test_confirmation_classification():
    """Test confirmation classification with variations"""
    print("\n🧪 TESTING CONFIRMATION CLASSIFICATION")
    print("=" * 60)
    
    test_cases = [
        ("yes", "confirm", "Should confirm with yes"),
        ("yessssss", "confirm", "Should handle repeated letters"),
        ("syre", "confirm", "Should handle typo for 'sure'"),
        ("sure", "confirm", "Should confirm with sure"),
        ("ok", "confirm", "Should confirm with ok"),
        ("no", "decline", "Should decline with no"),
        ("nope", "decline", "Should decline with nope"),
        ("what about balance", "unclear", "Should be unclear for different topic"),
        ("transfer 100 to Paul", "unclear", "Should be unclear for new transaction"),
        ("maybe later", "decline", "Should decline for maybe later"),
    ]
    
    state = TransferState()
    state['user_token'] = 'test_user'
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (test_input, expected_classification, description) in enumerate(test_cases, 1):
        print(f"\nTest {i:2d}: '{test_input}'")
        print(f"        Expected: {expected_classification}")
        print(f"        Description: {description}")
        
        try:
            result = classify_user_confirmation(test_input, state)
            actual_classification = result.get('classification', 'unknown')
            confidence = result.get('confidence', 0)
            
            if actual_classification == expected_classification:
                print(f"        ✅ PASS: {actual_classification} (confidence: {confidence}%)")
                success_count += 1
            else:
                print(f"        ❌ FAIL: Got {actual_classification}, expected {expected_classification}")
                print(f"           Confidence: {confidence}%")
                
        except Exception as e:
            print(f"        ❌ ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Confirmation Classification Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    return success_count, total_tests

def test_transaction_state_cancellation():
    """Test transaction state clearing on cancellation"""
    print("\n🧪 TESTING TRANSACTION STATE CANCELLATION")
    print("=" * 60)
    
    # Set up state with ongoing transaction
    state = TransferState()
    state['user_token'] = 'test_user'
    state['chat_history'] = []
    state['current_account'] = 'John'
    state['current_amount'] = 500
    
    print("Initial state:")
    print(f"  - Current account: {state.get('current_account')}")
    print(f"  - Current amount: {state.get('current_amount')}")
    
    cancellation_phrases = [
        "cancel that",
        "cancel", 
        "stop",
        "never mind",
        "abort"
    ]
    
    for phrase in cancellation_phrases:
        print(f"\nTesting cancellation with: '{phrase}'")
        try:
            # Reset state for each test
            state['current_account'] = 'John'
            state['current_amount'] = 500
            
            response = handle_general_inquiry(phrase, 'test_user', state)
            
            # Check if state was cleared
            if not state.get('current_account') and not state.get('current_amount'):
                print(f"        ✅ PASS: Transaction state cleared")
                print(f"        Response: {response}")
            else:
                print(f"        ❌ FAIL: Transaction state not cleared")
                print(f"        Account: {state.get('current_account')}")
                print(f"        Amount: {state.get('current_amount')}")
                
        except Exception as e:
            print(f"        ❌ ERROR: {e}")

def run_all_tests():
    """Run all classification tests"""
    print("🚀 ENHANCED CLASSIFICATION TESTING SUITE")
    print("=" * 80)
    print("Testing the improved LLM classification system for banking agent")
    print("=" * 80)
    
    total_success = 0
    total_tests = 0
    
    # Run all test suites
    success1, tests1 = test_intent_classification()
    total_success += success1
    total_tests += tests1
    
    success2, tests2 = test_general_inquiry_handling()
    total_success += success2
    total_tests += tests2
    
    success3, tests3 = test_confirmation_classification()
    total_success += success3
    total_tests += tests3
    
    # Test transaction state cancellation (no scoring)
    test_transaction_state_cancellation()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_success}/{total_tests}")
    print(f"Overall Success Rate: {(total_success/total_tests)*100:.1f}%")
    
    if total_success == total_tests:
        print("🎉 ALL TESTS PASSED! Classification system is working perfectly!")
    elif (total_success/total_tests) >= 0.9:
        print("✅ EXCELLENT! Classification system is working very well!")
    elif (total_success/total_tests) >= 0.8:
        print("👍 GOOD! Classification system is working well with minor issues!")
    else:
        print("⚠️  NEEDS IMPROVEMENT! Classification system has significant issues!")
    
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
