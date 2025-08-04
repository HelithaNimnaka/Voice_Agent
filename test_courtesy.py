#!/usr/bin/env python3
"""Test script for courtesy expressions and general inquiry handling."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_courtesy_expressions():
    """Test handling of courtesy expressions like thanks, appreciation, etc."""
    print("🧪 Testing courtesy expressions...")
    
    try:
        from graph_agent import execute_banking_graph
        
        test_cases = [
            "thanks",
            "thank you",
            "that was helpful",
            "great service",
            "appreciate it",
            "awesome",
            "perfect",
            "you're wonderful"
        ]
        
        results = []
        
        for i, test_input in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_input}'")
            result = execute_banking_graph(
                thread_id=f"test_courtesy_{i}",
                user_input=test_input,
                user_token="test_user",
                current_state=None
            )
            
            response = result.get('message', 'No response')
            print(f"   Response: {response}")
            
            # Check if response is appropriate (not redirecting to banking functions only)
            if "you're welcome" in response.lower() or "anything else" in response.lower():
                results.append(True)
                print("   ✅ Appropriate courtesy response")
            else:
                results.append(False)
                print("   ❌ Inappropriate response for courtesy expression")
        
        passed = sum(results)
        total = len(results)
        print(f"\n📋 Courtesy Expression Results: {passed}/{total} passed")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Error during courtesy test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_still_redirects_non_banking():
    """Test that non-banking questions still get redirected."""
    print("\n🧪 Testing non-banking redirections...")
    
    try:
        from graph_agent import execute_banking_graph
        
        test_cases = [
            "what's the weather like",
            "how are you doing",
            "tell me a joke",
            "what time is it"
        ]
        
        results = []
        
        for i, test_input in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_input}'")
            result = execute_banking_graph(
                thread_id=f"test_redirect_{i}",
                user_input=test_input,
                user_token="test_user", 
                current_state=None
            )
            
            response = result.get('message', 'No response')
            print(f"   Response: {response}")
            
            # Check if response redirects to banking functions
            if "banking functions" in response.lower() and "transfers" in response.lower():
                results.append(True)
                print("   ✅ Properly redirected to banking")
            else:
                results.append(False)
                print("   ❌ Should have redirected to banking")
        
        passed = sum(results)
        total = len(results)
        print(f"\n📋 Redirection Results: {passed}/{total} passed")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Error during redirection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all courtesy and general inquiry tests."""
    print("🚀 Testing Courtesy & General Inquiry Handling\n")
    
    # Test courtesy expressions
    courtesy_result = test_courtesy_expressions()
    
    # Test non-banking redirections still work
    redirect_result = test_still_redirects_non_banking()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   Courtesy Expressions: {'✅ PASSED' if courtesy_result else '❌ FAILED'}")
    print(f"   Non-Banking Redirections: {'✅ PASSED' if redirect_result else '❌ FAILED'}")
    
    if courtesy_result and redirect_result:
        print("\n🎉 All courtesy handling tests passed!")
    else:
        print("\n⚠️ Some courtesy handling tests failed.")

if __name__ == "__main__":
    main()
