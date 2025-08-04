#!/usr/bin/env python3
"""Test script for LLM-based courtesy and general inquiry handling."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_llm_based_classification():
    """Test LLM-based classification for various user expressions."""
    print("🧪 Testing LLM-based general inquiry classification...")
    
    try:
        from graph_agent import execute_banking_graph
        
        test_cases = [
            # Greetings
            ("helloo", "greeting"),
            ("good morning", "greeting"),
            ("hey there", "greeting"),
            
            # Courtesy expressions
            ("life server!!!", "courtesy"),
            ("you're amazing", "courtesy"),
            ("that was super helpful", "courtesy"),
            ("thanks a lot", "courtesy"),
            ("appreciate your help", "courtesy"),
            
            # Cancellation requests
            ("no no cancel that", "cancellation"),
            ("stop this", "cancellation"),
            ("cancel please", "cancellation"),
            ("abort", "cancellation"),
            
            # Other non-banking queries
            ("what about LB finance", "other"),
            ("tell me about your services", "other"),
            ("what are the rates", "other"),
            ("where are your branches", "other")
        ]
        
        results = []
        
        for i, (test_input, expected_type) in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_input}' (Expected: {expected_type})")
            result = execute_banking_graph(
                thread_id=f"test_llm_{i}",
                user_input=test_input,
                user_token="test_user",
                current_state=None
            )
            
            response = result.get('message', 'No response')
            print(f"   Response: {response}")
            
            # Check if response matches expected behavior
            success = False
            if expected_type == "greeting" and ("Hello!" in response or "How can I help" in response):
                success = True
            elif expected_type == "courtesy" and "You're welcome" in response:
                success = True
            elif expected_type == "cancellation" and "Understood" in response:
                success = True
            elif expected_type == "other" and "banking functions" in response:
                success = True
            
            results.append(success)
            print(f"   {'✅ Correct classification' if success else '❌ Incorrect classification'}")
        
        passed = sum(results)
        total = len(results)
        print(f"\n📋 LLM Classification Results: {passed}/{total} passed")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Error during LLM classification test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flexible_expressions():
    """Test how well the LLM handles flexible, creative expressions."""
    print("\n🧪 Testing flexible expression handling...")
    
    try:
        from graph_agent import execute_banking_graph
        
        creative_expressions = [
            "you're a lifesaver!",
            "omg thank youuu",
            "super duper helpful",
            "you rock!",
            "this is brilliant",
            "couldn't be better",
            "exactly what I needed",
            "nah cancel this whole thing",
            "forget about it",
            "never mind stop"
        ]
        
        results = []
        
        for i, test_input in enumerate(creative_expressions):
            print(f"\n{i+1}. Testing flexible expression: '{test_input}'")
            result = execute_banking_graph(
                thread_id=f"test_flex_{i}",
                user_input=test_input,
                user_token="test_user",
                current_state=None
            )
            
            response = result.get('message', 'No response')
            print(f"   Response: {response}")
            
            # Check if it's handled appropriately (not just redirected to banking)
            is_appropriate = (
                "You're welcome" in response or 
                "Understood" in response or 
                "How can I help" in response or
                ("banking functions" in response and any(word in test_input.lower() for word in ["cancel", "forget", "stop", "never mind"]))
            )
            
            results.append(is_appropriate)
            print(f"   {'✅ Handled appropriately' if is_appropriate else '❌ Not handled well'}")
        
        passed = sum(results)
        total = len(results)
        print(f"\n📋 Flexible Expression Results: {passed}/{total} handled appropriately")
        
        return passed >= total * 0.8  # 80% success rate is good for flexible expressions
        
    except Exception as e:
        print(f"❌ Error during flexible expression test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all LLM-based classification tests."""
    print("🚀 Testing LLM-Based General Inquiry Handling\n")
    
    # Test LLM classification
    classification_result = test_llm_based_classification()
    
    # Test flexible expressions
    flexible_result = test_flexible_expressions()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   LLM Classification: {'✅ PASSED' if classification_result else '❌ FAILED'}")
    print(f"   Flexible Expressions: {'✅ PASSED' if flexible_result else '❌ FAILED'}")
    
    if classification_result and flexible_result:
        print("\n🎉 All LLM-based classification tests passed!")
    else:
        print("\n⚠️ Some LLM-based classification tests failed.")

if __name__ == "__main__":
    main()
