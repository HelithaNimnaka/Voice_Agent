import os
import json
import re
import requests
import datetime
from dotenv import load_dotenv
import os
import json
import re
import datetime
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.chat_models import init_chat_model

from tools.checkAccountBalance import CheckAccountBalance
from tools.checkAccountExistence import CheckAccountExistence
from tools.processTransfer import ProcessTransfer

# ───────────────────────────────────────────────────────────────
# Graph Agent for LB Finance Banking Bot
# ───────────────────────────────────────────────────────────────
load_dotenv(override=True)

# Load the API key securely - check if we're in Streamlit context
try:
    import streamlit as st
    groq_api_key = st.secrets["GROQ_API_KEY"]
    print("✅ Loaded API key from Streamlit secrets")
except:
    # Fallback to environment variable if not in Streamlit
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    print("✅ Loaded API key from environment variable")

if not groq_api_key:
    print("⚠️ WARNING: No GROQ_API_KEY found! LLM will not work properly.")

try:
    llm = init_chat_model(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        model_provider="groq", 
        api_key=groq_api_key,
        temperature=0.3
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize LLM: {e}")
    llm = None

from functools import lru_cache

@lru_cache(maxsize=None)
def get_combined_agent():
    tools = [
        CheckAccountExistence(),
        CheckAccountBalance(),
        ProcessTransfer()
    ]
    return create_react_agent(llm, tools=tools)

class ConversationMessage(TypedDict):
    """Single conversation message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    turn_number: int

class TransferState(TypedDict):
    # User Profile Information (Frontend & Agent Shared)
    user_name: str
    user_account: str
    user_balance: str
    user_status: str
    
    # Transaction Flow Data
    user_query: str
    destination_account: str
    transfer_amount: str
    user_token: str
    ai_response: str
    transaction_result: str
    
    # Confirmation Management
    needs_confirmation: bool
    confirmation_requested: bool
    user_confirmed: bool
    
    # Memory & Conversation Management
    chat_history: List[ConversationMessage]
    conversation_context: str  # Summary of ongoing transaction context
    turn_number: int
    thread_id: str
    
    # State Management Flags
    profile_loaded: bool
    last_balance_check: str  # Timestamp to avoid frequent DB calls

# ───────────────────────────────────────────────────────────────
# MEMORY SERVER & CONVERSATION MANAGEMENT
# ───────────────────────────────────────────────────────────────

class ConversationMemoryServer:
    """Manages conversation memory and context across turns"""
    
    def __init__(self):
        self.memory_store = {}  # In-memory store for conversation histories
    
    def get_conversation_context(self, thread_id: str, current_state: TransferState) -> str:
        """Generate conversation context summary for the agent"""
        chat_history = current_state.get('chat_history', [])
        
        if not chat_history:
            return "This is the start of a new conversation."
        
        # Check if we have any ongoing transaction data
        has_ongoing_transaction = bool(
            current_state.get('destination_account') and current_state.get('destination_account') not in [None, "None", ""] or
            current_state.get('transfer_amount') and current_state.get('transfer_amount') not in [None, "None", ""]
        )
        
        if not has_ongoing_transaction:
            # If no ongoing transaction, don't include transaction details from history
            return "No ongoing transaction. User is starting a new interaction."
        
        # Only include recent context if there's an ongoing transaction
        recent_messages = chat_history[-3:] if len(chat_history) > 3 else chat_history
        
        context_summary = "RECENT CONVERSATION CONTEXT:\n"
        for msg in recent_messages:
            context_summary += f"- {msg['role'].title()}: {msg['content']}\n"
        
        # Add transaction context if any
        if current_state.get('destination_account') or current_state.get('transfer_amount'):
            context_summary += f"\nONGOING TRANSACTION:\n"
            if current_state.get('destination_account'):
                context_summary += f"- Destination: {current_state['destination_account']}\n"
            if current_state.get('transfer_amount'):
                context_summary += f"- Amount: ${current_state['transfer_amount']}\n"
        
        return context_summary
    
    def add_message_to_history(self, state: TransferState, role: str, content: str) -> TransferState:
        """Add a new message to conversation history"""
        if 'chat_history' not in state:
            state['chat_history'] = []
        
        new_message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.datetime.now().isoformat(),
            turn_number=state.get('turn_number', 0) + 1
        )
        
        state['chat_history'].append(new_message)
        state['turn_number'] = new_message['turn_number']
        
        return state
    
    def update_conversation_context(self, state: TransferState) -> TransferState:
        """Update conversation context based on current transaction state"""
        context_parts = []
        
        if state.get('destination_account') and state.get('destination_account') not in [None, "None", ""]:
            context_parts.append(f"Transferring to: {state['destination_account']}")
        
        if state.get('transfer_amount') and state.get('transfer_amount') not in [None, "None", ""]:
            context_parts.append(f"Amount: ${state['transfer_amount']}")
        
        state['conversation_context'] = " | ".join(context_parts) if context_parts else ""
        
        return state

# Global memory server instance
memory_server = ConversationMemoryServer()

def get_user_primary_account(user_token: str) -> str:
    """Get user's primary account from database."""
    try:
        from firebase import DatabaseManager
        db_manager = DatabaseManager("main_DB")
        account = db_manager.get_user_primary_account(user_token)
        print(f"DEBUG: Primary account for {user_token}: {account}")
        return account
    except Exception as e:
        print(f"DEBUG: Error getting primary account: {e}")
        return "admin"  # Default fallback

def load_user_profile(user_token: str, state: TransferState) -> TransferState:
    """Load complete user profile information into state from actual database."""
    print(f"🔍 LOADING USER PROFILE: {user_token}")
    
    # Check if profile already loaded recently (avoid redundant DB calls)
    import datetime
    current_time = datetime.datetime.now().isoformat()
    last_check = state.get("last_balance_check", "")
    
    if state.get("profile_loaded") and last_check:
        try:
            last_time = datetime.datetime.fromisoformat(last_check)
            time_diff = datetime.datetime.now() - last_time
            if time_diff.seconds < 300:  # 5 minutes cache
                print(f"✅ Using cached profile data (last updated: {time_diff.seconds}s ago)")
                return state
        except:
            pass  # If parsing fails, reload profile
    
    try:
        from firebase import DatabaseManager
        db_manager = DatabaseManager("main_DB")
        
        # Get user details using available methods
        user_data = db_manager.get_full_user_data(user_token)
        user_account = db_manager.get_user_primary_account(user_token)
        user_balance = db_manager.get_account_balance(user_token)
        payees = db_manager.get_payee_list(user_token)
        
        if user_data:
            # Update state with real data from database
            db_name = user_data.get("Name", "User Name")
            # Override generic "User ###" names with "User Name"
            if db_name and db_name.startswith("User ") and db_name.split()[-1].isdigit():
                state["user_name"] = "User Name"
            else:
                state["user_name"] = db_name
            state["user_account"] = user_data.get("Account Number", user_account or "Unknown")
            state["user_balance"] = user_data.get("Account Balance", f"${user_balance}" if user_balance else "$0.00")
            state["user_status"] = "Active"  # Default status
            state["profile_loaded"] = True
            state["last_balance_check"] = current_time
            
            print(f"✅ Profile loaded from database: {state['user_name']}, Balance: {state['user_balance']}")
        else:
            # If no user data found, load what we can
            state["user_name"] = "User Name"
            state["user_account"] = user_account or "Unknown"
            state["user_balance"] = f"${user_balance}" if user_balance else "$0.00"
            state["user_status"] = "Active"
            state["profile_loaded"] = True
            state["last_balance_check"] = current_time
            
            print(f"✅ Profile loaded with available data: {state['user_name']}")
        
    except Exception as e:
        print(f"⚠️ Error loading profile from DB: {e}")
        # Set minimal profile information to avoid errors
        state["user_name"] = "User Name"
        state["user_account"] = "Unknown"
        state["user_balance"] = "$0.00"
        state["user_status"] = "Error loading profile"
        state["profile_loaded"] = False
        state["last_balance_check"] = current_time
        
        print(f"❌ Failed to load profile, using minimal data")
    
    return state

def get_user_payees_list(user_token: str) -> str:
    """Get formatted payees list for display from actual database."""
    try:
        from firebase import DatabaseManager
        db_manager = DatabaseManager("main_DB")
        payees = db_manager.get_payee_list(user_token)
        
        if payees:
            # Handle different payee data formats
            if isinstance(payees, str):
                # If it's already a string, return as is
                return payees
            elif isinstance(payees, list):
                # If it's a list, join with commas
                return ", ".join(str(payee) for payee in payees)
            else:
                return "No payees found"
        else:
            return "No payees found"
    except Exception as e:
        print(f"DEBUG: Error getting payees: {e}")
        return "No payees found"

def get_comprehensive_user_context(state: TransferState) -> str:
    """Generate comprehensive user context including profile, payees, chat history, and current transaction state."""
    user_token = state.get("user_token", "")
    
    # Load fresh user data
    try:
        from firebase import DatabaseManager
        db_manager = DatabaseManager("main_DB")
        user_data = db_manager.get_full_user_data(user_token)
        payees = db_manager.get_payee_list(user_token)
        balance = db_manager.get_account_balance(user_token)
        account = db_manager.get_user_primary_account(user_token)
        
        # Format payees list
        payees_str = "None"
        if payees:
            if isinstance(payees, list):
                payees_str = ", ".join(payees)
            elif isinstance(payees, str):
                payees_str = payees
        
        # Build comprehensive context
        context = f"""
CURRENT USER PROFILE:
- User Token: {user_token}
- Name: {user_data.get('Name', 'Unknown') if user_data else 'Unknown'}
- Account Number: {account or 'Unknown'}
- Current Balance: ${balance or '0.00'}
- Status: Active
- Available Payees: {payees_str}

IMPORTANT: The payee list and balance are for system use only. DO NOT mention specific payee names or balance amounts in responses unless the user specifically asks for this information.

CHAT HISTORY:
"""
        
        # Add chat history if available
        chat_history = state.get('chat_history', [])
        if chat_history:
            # Show last 5 messages for context
            recent_messages = chat_history[-5:] if len(chat_history) > 5 else chat_history
            for msg in recent_messages:
                context += f"- {msg['role'].title()}: {msg['content']}\n"
        else:
            context += "- No previous conversation\n"
        
        # Add current transaction state
        context += f"""
CURRENT TRANSACTION STATE:
- Destination Account: {state.get('destination_account', 'Not set')}
- Transfer Amount: {state.get('transfer_amount', 'Not set')}
- Transaction Status: {state.get('transaction_result', 'Not started')}
- Last Response: {state.get('ai_response', 'None')}

CONVERSATION CONTEXT: {state.get('conversation_context', 'No ongoing transaction')}
"""
        
        return context
        
    except Exception as e:
        print(f"Error building user context: {e}")
        return f"""
CURRENT USER PROFILE:
- User Token: {user_token}
- Error loading profile: {str(e)}
- Status: Error

CHAT HISTORY:
- Unable to load chat history

CURRENT TRANSACTION STATE:
- Error loading transaction state
"""

def analyze_user_intent_and_extract_data(user_query: str, user_token: str, state: TransferState) -> dict:
    """Use NOVA team approach to analyze intent and extract transaction data with full user context."""
    print(f"🧠 NOVA Team analyzing: {user_query}")

    # Get comprehensive user context
    full_context = get_comprehensive_user_context(state)
    
    # Check if we have ongoing transaction context
    has_ongoing_context = bool(
        state.get('destination_account') and state.get('destination_account') not in [None, "None", ""] or
        state.get('transfer_amount') and state.get('transfer_amount') not in [None, "None", ""]
    )
    
    context_instruction = ""
    if not has_ongoing_context:
        context_instruction = """
IMPORTANT: No ongoing transaction detected. Analyze ONLY the current user input. If the user mentions only an amount (like "300"), do NOT assume any previous recipient names unless they are explicitly mentioned in the current input.
"""
    else:
        context_instruction = """
IMPORTANT: There is an ongoing transaction. Consider the context for continuation.
"""

    prompt = f"""
You are NOVA — a team of expert AI assistants collaborating to analyze user intent and extract banking information.

{full_context}

{context_instruction}

CURRENT USER INPUT: "{user_query}"

Your task is to analyze the user's intent while considering the full user context and conversation history above.

Intent Categories:
1. "transaction" - Money transfer related (send, transfer, pay, etc.) BUT NOT cancellation requests
2. "payee_list" - Asking about saved payees/contacts
3. "balance_inquiry" - Asking about account balance or balance-related questions
4. "compound" - Multiple questions or requests in one input (e.g., payees + balance)
5. "add_payee" - Requests to add new payees/accounts (e.g., "add new account", "add payee", "need to add")
6. "general" - ONLY greetings and non-banking queries that should be redirected to banking functions

SPECIAL HANDLING FOR CANCELLATION:
- If user says "cancel", "cancel that", "stop", "abort", "never mind" during ANY context, classify as "general" with cancellation intent
- Cancellation requests should CLEAR ongoing transaction state and reset the conversation

IMPORTANT RESTRICTION: This agent ONLY handles core banking functions:
- Money transfers/transactions
- Account balance inquiries  
- Payee list management
- Greetings (redirected to banking functions)

ALL other queries (rates, loans, services, products, branches, etc.) should be classified as "general" and redirected.

Transaction Data Extraction (ONLY for transaction intent):
- account: Extract payee name/account if explicitly mentioned in current input (None if not mentioned)
  * IMPORTANT: Use intelligent understanding to detect when user is correcting/changing payee
  * Examples: "no no to Sarah", "not Mike, to Sarah", "actually to Bob", "change to Alice", "no no to Mike"
  * Pattern recognition: "no no to [NAME]" = user wants to change to [NAME]
  * If user mentions a different payee than currently set, they likely want to change it
  * Trust natural language understanding over pattern matching
  * CRITICAL: For corrections like "no no to Mike", extract "Mike" as the account
- amount: Extract monetary amount as integer (null if not mentioned)
  * IMPORTANT: Use intelligent understanding to detect when user is correcting/changing amount
  * Examples: "no 200", "not 500, make it 300", "actually 150", "change it to 100", "sorry 250", "I meant 400"
  * Also detect: "no wait, 500", "make that 200", "correct it to 300", "update to 600"
  * If user mentions a different amount than currently set, they likely want to change it
  * Trust natural language understanding over pattern matching

Context Awareness Rules:
- Use the user's actual payee list shown above for validation
- Consider chat history for context but extract ONLY what is explicitly mentioned in current input
- If NO ongoing transaction: Extract ONLY what is explicitly mentioned in current input
- If ongoing transaction exists: Consider context for missing information, and use intelligent understanding for corrections
- DO NOT infer account names from completed transactions in chat history
- Amount-only inputs (like "300") should have account="None" unless there's an ongoing transaction
- Use natural language understanding to detect corrections - don't rely on specific keywords
- If user mentions different account/amount than current state, they likely want to change it
- Trust context and intent over rigid pattern matching

TEAM ROLES:

1. Intent Analyzer
   - Determines user intent from these categories:
     * "transaction": transfers, payments, send money, topup, account operations, explicit requests to move money
     * "payee_list": requests to view/show payee list, saved accounts, beneficiaries
     * "balance_inquiry": requests about account balance, current balance, balance check
     * "compound": multiple questions in one input (e.g., "who are my payees and what is my balance")
     * "add_payee": requests to add new payees/accounts (e.g., "add new account", "add payee", "need to add", "add new", "I want to add")
     * "general": ONLY greetings and non-banking queries (rates, products, branches, services, etc.) - ALL should be redirected to banking functions
   - PRIORITY: Detect compound questions that ask for multiple things at once
   - Examples of BALANCE_INQUIRY intent: "what is my balance", "check balance", "account balance", "how much do I have"
   - Examples of COMPOUND intent: "who are my payees and what is my balance", "show payees and balance", "payees list and current balance"
   - Examples of ADD_PAYEE intent: "add new account", "add payee", "need to add", "I want to add new payee", "add new account to my list", "how to add payee"
   - Examples of GENERAL intent: "hello", "hi", "FD rates", "loan rates", "branches", "services", "about LB Finance", "help", "how are you", "weather", "news" - ALL redirected to banking functions
   - Examples of TRANSACTION intent: "send", "transfer", "pay", "to [payee]", "let's do a transaction", "shall we do a transaction", "move money", "do a transfer", "start a transaction", standalone amounts with ongoing transaction context

2. Data Extractor  
   - Extracts account/payee names if explicitly mentioned in current input
   - Validates extracted account names against the user's actual payee list shown above
   - Extracts amounts (like "500", "$1000", "200 rs")
   - Uses intelligent natural language understanding to detect corrections and changes
   - If user mentions different payee/amount in context of existing transaction, treats as correction
   - Considers conversational flow and user intent rather than rigid keyword matching
   - Examples of corrections: "no no to Sarah" (change to Sarah), "make it 200" (change amount to 200)

3. Schema Enforcer
   - Ensures output follows this exact JSON format:
   {{
     "intent": "transaction" | "payee_list" | "balance_inquiry" | "compound" | "add_payee" | "general",
     "account": "<name>" | "None",
     "amount": <number> | null,
     "confidence": <0-100>,
     "user_token": "{user_token}"
   }}

4. Quality Controller
   - Validates the analysis follows the context rules
   - Ensures account names match the user's actual payee list
   - Returns only the final JSON object (no explanations)

Examples:
- "Send 500 to Alice" → {{"intent": "transaction", "account": "Alice", "amount": 500, "confidence": 95, "user_token": "{user_token}"}}
- "Transfer to Bob" → {{"intent": "transaction", "account": "Bob", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "300" (no ongoing transaction) → {{"intent": "transaction", "account": "None", "amount": 300, "confidence": 80, "user_token": "{user_token}"}}
- "no no to Sarah" → {{"intent": "transaction", "account": "Sarah", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "no no to Mike" → {{"intent": "transaction", "account": "Mike", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "not Mike, to Sarah" → {{"intent": "transaction", "account": "Sarah", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "actually to Bob" → {{"intent": "transaction", "account": "Bob", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "make it Sarah instead" → {{"intent": "transaction", "account": "Sarah", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "no 200" → {{"intent": "transaction", "account": "None", "amount": 200, "confidence": 90, "user_token": "{user_token}"}}
- "not 500, make it 300" → {{"intent": "transaction", "account": "None", "amount": 300, "confidence": 90, "user_token": "{user_token}"}}
- "actually 150" → {{"intent": "transaction", "account": "None", "amount": 150, "confidence": 85, "user_token": "{user_token}"}}
- "change it to 100" → {{"intent": "transaction", "account": "None", "amount": 100, "confidence": 85, "user_token": "{user_token}"}}
- "sorry 250" → {{"intent": "transaction", "account": "None", "amount": 250, "confidence": 85, "user_token": "{user_token}"}}
- "I meant 400" → {{"intent": "transaction", "account": "None", "amount": 400, "confidence": 85, "user_token": "{user_token}"}}
- "no wait, 500" → {{"intent": "transaction", "account": "None", "amount": 500, "confidence": 85, "user_token": "{user_token}"}}
- "make that 200" → {{"intent": "transaction", "account": "None", "amount": 200, "confidence": 85, "user_token": "{user_token}"}}
- "correct it to 300" → {{"intent": "transaction", "account": "None", "amount": 300, "confidence": 85, "user_token": "{user_token}"}}
- "update to 600" → {{"intent": "transaction", "account": "None", "amount": 600, "confidence": 85, "user_token": "{user_token}"}}
- "who are my payees" → {{"intent": "payee_list", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "what is my balance" → {{"intent": "balance_inquiry", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "check my account balance" → {{"intent": "balance_inquiry", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "who are my payees and what is my balance" → {{"intent": "compound", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "show payees and current balance" → {{"intent": "compound", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "add new account" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "add payee" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "need to add" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "I want to add new payee" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "how to add payee" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "yes need to add" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "add new account to my list" → {{"intent": "add_payee", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "move money" → {{"intent": "transaction", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "do a transfer" → {{"intent": "transaction", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "let's do a transaction" → {{"intent": "transaction", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "shall we do a transaction" → {{"intent": "transaction", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "start a transaction" → {{"intent": "transaction", "account": "None", "amount": null, "confidence": 85, "user_token": "{user_token}"}}
- "cancel that" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "cancel" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "stop" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "never mind" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "abort" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "hello" → {{"intent": "general", "account": "None", "amount": null, "confidence": 95, "user_token": "{user_token}"}}
- "thanks buddy" → {{"intent": "general", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "that's all" → {{"intent": "general", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "we're done" → {{"intent": "general", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}
- "all good" → {{"intent": "general", "account": "None", "amount": null, "confidence": 90, "user_token": "{user_token}"}}

CRITICAL RULES:
- This agent ONLY handles core banking functions: transfers, balance checks, payee management
- Questions about rates, services, products, branches, loans, general inquiries = ALWAYS "general" intent 
- ALL non-banking queries should be classified as "general" and redirected to banking functions
- Banking service inquiries are NOT transactions
- Only explicit transfer/payment language should trigger "transaction" intent
- Greetings should be handled appropriately but redirected to banking functions
- Use intelligent natural language understanding for corrections and changes
- If user mentions different payee/amount in ongoing transaction context, extract it
- Trust conversational context and user intent over rigid pattern matching
- Be flexible with language - users express corrections in many different ways

Output valid JSON only:
{{"intent": "transaction/payee_list/balance_inquiry/compound/add_payee/general", "account": "name_or_None", "amount": number_or_null, "confidence": 0-100, "user_token": "{user_token}"}}

User query: "{user_query}"

User query: "{user_query}"
"""

    messages = [SystemMessage(content=prompt), HumanMessage(content=user_query)]

    try:
        if llm is None:
            print("🧠 LLM not available, using fallback")
            raise Exception("LLM not initialized")
            
        response = llm.invoke(messages).content.strip()
        print(f"🧠 NOVA response: {response}")
        parsed = json.loads(response)
        
        if isinstance(parsed, dict):
            # Ensure all required fields exist
            parsed.setdefault("intent", "general")
            parsed.setdefault("account", "None")
            parsed.setdefault("amount", None)
            parsed.setdefault("confidence", 50)
            parsed["user_token"] = user_token
            return parsed
            
    except Exception as e:
        print(f"🧠 NOVA error: {e}")
        
        # Fallback: Use rule-based classification when LLM fails
        print("🔧 Using fallback rule-based classification")
        intent = "general"
        confidence = 70
        
        # Simple rule-based classification
        query_lower = user_query.lower()
        
        # Get user's actual payee list for context-aware classification
        try:
            payee_list = get_user_payees_list(user_token)
            user_payees = []
            if payee_list and payee_list != "No payees found":
                # Parse payee list - handle both comma-separated and space-separated
                import re
                # Split by comma or and, then clean up
                payee_names = re.split(r'[,&]|\sand\s', payee_list.lower())
                user_payees = [name.strip() for name in payee_names if name.strip()]
                print(f"🔧 User payees for context: {user_payees}")
        except Exception as pe:
            print(f"🔧 Error getting payees for fallback: {pe}")
            user_payees = []
        
        # Check if we're in transaction continuation mode
        has_ongoing_transaction = bool(
            state.get('destination_account') and state.get('destination_account') not in [None, "None", ""] or
            state.get('transfer_amount') and state.get('transfer_amount') not in [None, "None", ""]
        )
        
        # Check for compound questions (multiple requests)
        compound_indicators = [
            ("balance" in query_lower or "account" in query_lower) and ("payee" in query_lower or "list" in query_lower),
            ("check" in query_lower) and ("tell" in query_lower),
            " and " in query_lower and any(x in query_lower for x in ["balance", "payee", "list", "account"])
        ]
        
        if any(compound_indicators):
            intent = "compound"
            confidence = 85
            print("🔧 Fallback: Detected compound question")
        
        # Check if input matches a known payee name (for ongoing transactions)
        elif (has_ongoing_transaction and user_payees and 
              any(payee in query_lower for payee in user_payees)):
            intent = "transaction"
            confidence = 90
            print(f"🔧 Fallback: Detected payee name in ongoing transaction context: {user_query}")
        
        # Check for specific intents - BANKING FUNCTIONS ONLY
        elif any(x in query_lower for x in ["balance", "account balance", "how much", "current balance"]):
            intent = "balance_inquiry"
            confidence = 80
            print("🔧 Fallback: Detected balance inquiry")
        
        elif any(x in query_lower for x in ["payee", "payees", "list", "contacts", "recipients"]):
            intent = "payee_list"
            confidence = 80
            print("🔧 Fallback: Detected payee list request")
        
        elif any(x in query_lower for x in ["add new", "add payee", "add account", "need to add", "want to add", "how to add"]):
            intent = "add_payee"
            confidence = 85
            print("🔧 Fallback: Detected add payee request")
        
        elif any(x in query_lower for x in ["transfer", "send", "pay", "money", "transaction", "do a", "let's do", "shall we", "move money"]) or query_lower.strip().isdigit():
            intent = "transaction"
            confidence = 75
            print("🔧 Fallback: Detected transaction")
        
        # Special case: if we have ongoing transaction and user provides just a number, treat as transaction
        elif has_ongoing_transaction and query_lower.strip().isdigit():
            intent = "transaction"
            confidence = 85
            print(f"🔧 Fallback: Number input in transaction context: {user_query}")
        
        # Check for non-banking queries (rates, services, products, etc.) - classify as general
        elif any(x in query_lower for x in ["rate", "rates", "loan", "fd", "fixed deposit", "branch", "service", "services", "product", "products", "about", "weather", "news", "how are you"]):
            intent = "general"
            confidence = 90
            print("🔧 Fallback: Detected non-banking query, classified as general")
        
        # If we have an ongoing transaction and the input is short (likely a payee name or amount)
        elif has_ongoing_transaction and len(user_query.strip().split()) <= 2:
            intent = "transaction"
            confidence = 80
            print(f"🔧 Fallback: Short input in transaction context, treating as transaction: {user_query}")
        
        # Default: anything else is general (non-banking)
        else:
            intent = "general"
            confidence = 70
            print("🔧 Fallback: Default classification as general (non-banking query)")
        
        # Extract basic transaction data
        account_extracted = "None"
        amount_extracted = None
        
        # Enhanced account extraction - check against known payees
        if user_payees:
            for payee in user_payees:
                if payee in query_lower:
                    # Find the original case version from the payee list
                    original_payee_list = get_user_payees_list(user_token)
                    if original_payee_list:
                        import re
                        payee_matches = re.findall(rf'\b({re.escape(payee)})\b', original_payee_list, re.IGNORECASE)
                        if payee_matches:
                            account_extracted = payee_matches[0]
                            print(f"🔧 Fallback: Extracted payee account: {account_extracted}")
                            break
        
        # Simple amount extraction
        import re
        amount_match = re.search(r'\b(\d+(?:\.\d{2})?)\b', user_query)
        if amount_match:
            amount_extracted = int(float(amount_match.group(1)))
            print(f"🔧 Fallback: Extracted amount: {amount_extracted}")
        
        return {
            "intent": intent,
            "account": account_extracted, 
            "amount": amount_extracted,
            "confidence": confidence,
            "user_token": user_token
        }

def handle_general_inquiry(user_query: str, user_token: str, state: TransferState) -> str:
    """Handle greetings and courtesy expressions using intelligent LLM classification."""
    user_query_lower = user_query.strip().lower()
    chat_history = state.get('chat_history', [])
    assistant_messages = [msg for msg in chat_history if msg.get('role') == 'assistant']
    is_first_interaction = len(assistant_messages) == 0
    
    # Use LLM to classify the type of general inquiry
    classification_prompt = f"""
You are an AI assistant that classifies user messages into categories. 

USER MESSAGE: "{user_query}"
CONVERSATION CONTEXT: {"First interaction" if is_first_interaction else "Ongoing conversation"}

Classify this message into ONE of these categories:

1. "greeting" - Initial greetings like "hi", "hello", "hey", "good morning", etc.
2. "courtesy" - Thanks, appreciation, compliments like "thanks", "appreciate it", "great service", "helpful", "awesome", "perfect", "you're wonderful", "life saver", "buddy", "thank you", "cheers", "that's all", "all good", "we're done", "I'm done", etc.
3. "cancellation" - User wants to cancel or stop something like "cancel", "no no cancel", "stop", "abort", etc.
4. "other" - Any other non-banking query that should be redirected to banking functions

Examples:
- "thanks buddy" → courtesy
- "thank you so much" → courtesy  
- "you're amazing" → courtesy
- "cheers mate" → courtesy
- "appreciate it" → courtesy
- "that's all" → courtesy
- "all good" → courtesy
- "we're done" → courtesy
- "cancel that" → cancellation
- "cancel" → cancellation
- "stop" → cancellation
- "never mind" → cancellation
- "abort" → cancellation

Return ONLY one word: greeting, courtesy, cancellation, or other
"""
    
    try:
        if llm is not None:
            response = llm.invoke(classification_prompt).content.strip().lower()
            classification = response if response in ["greeting", "courtesy", "cancellation", "other"] else "other"
        else:
            # Fallback to simple word matching if LLM unavailable
            if any(word in user_query_lower for word in ["hi", "hello", "hey", "morning", "afternoon", "evening"]):
                classification = "greeting"
            elif any(word in user_query_lower for word in ["thank", "thanks", "appreciate", "great", "awesome", "perfect", "helpful", "wonderful", "life", "saver", "buddy", "mate", "cheers", "all good", "done", "that's all"]):
                classification = "courtesy" 
            elif any(word in user_query_lower for word in ["cancel", "stop", "abort", "never mind", "no no"]):
                classification = "cancellation"
            else:
                classification = "other"
    except Exception as e:
        print(f"Error in LLM classification: {e}")
        # Enhanced fallback with better pattern matching
        if any(word in user_query_lower for word in ["hi", "hello", "hey", "morning", "afternoon", "evening"]):
            classification = "greeting"
        elif any(word in user_query_lower for word in ["thank", "thanks", "appreciate", "great", "awesome", "perfect", "helpful", "wonderful", "life", "saver", "buddy", "mate", "cheers", "all good", "done", "that's all"]):
            classification = "courtesy" 
        elif any(word in user_query_lower for word in ["cancel", "stop", "abort", "never mind", "no no"]):
            classification = "cancellation"
        else:
            classification = "other"
    
    print(f"🔍 General inquiry classification: {classification}")
    
    # Handle based on classification
    if classification == "greeting":
        if is_first_interaction:
            return "Hello! I'm LEO, your LB Finance banking assistant. I can help you with money transfers, checking account balance, and managing your payee list. How can I assist you today?"
        else:
            return "How can I help you with your banking needs?"
    
    elif classification == "courtesy":
        return "You're welcome! Is there anything else I can help you with regarding your banking needs?"
    
    elif classification == "cancellation":
        # Clear any ongoing transaction state - use correct field names
        transaction_cleared = False
        if (state.get('destination_account') and state.get('destination_account') not in [None, "None", ""]) or \
           (state.get('transfer_amount') and state.get('transfer_amount') not in [None, "None", ""]):
            state['destination_account'] = ""
            state['transfer_amount'] = ""
            state['needs_confirmation'] = False
            state['confirmation_requested'] = False
            state['user_confirmed'] = False
            transaction_cleared = True
            print("🔄 Cleared transaction state due to cancellation")
        
        if transaction_cleared:
            return "Transaction cancelled. How can I help you with your banking needs today?"
        else:
            return "How can I help you with your banking needs today?"
    
    else:  # "other" - non-banking queries
        return "I can only assist with banking functions like money transfers, checking your account balance, and managing your payee list. How can I help you with these services?"

def generate_welcome_message(state: TransferState) -> str:
    """Generate welcome message using NOVA team approach with full user context."""
    full_context = get_comprehensive_user_context(state)
    
    # Check if this is the very first interaction (no previous assistant responses)
    chat_history = state.get('chat_history', [])
    assistant_messages = [msg for msg in chat_history if msg.get('role') == 'assistant']
    is_first_interaction = len(assistant_messages) == 0  # No previous assistant responses
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user has started a transfer process. Create a personalized message that:
1. {"Welcome them by name (if available) since this is their first interaction with the assistant" if is_first_interaction else "Do NOT use greetings like 'Hi!' since there's ongoing conversation history"}
2. Asks for the payee name from their saved list (DO NOT list the specific payee names without being asked)
3. Asks for the transfer amount (No need to tell the account balance without being asked)
4. Keep it friendly, professional, and mobile-optimized (under 3 sentences)

CRITICAL RULES:
- DO NOT mention specific payee names in your response
- {"Include a brief greeting for first interactions" if is_first_interaction else "Avoid greetings in ongoing conversations"}
- Just ask them to provide the payee name from their saved list
- Keep it concise and professional

IMPORTANT: Return ONLY the message text that should be shown to the user. Do not include any explanations, analysis, or meta-commentary.

Example outputs:
{"- 'Hello! I'll help you with your transfer. Please provide the payee name from your saved list and the amount you'd like to transfer.'" if is_first_interaction else "- 'I'll help you with your transfer. Please provide the payee name from your saved list and the amount you'd like to transfer.'"}
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM welcome message: {e}")
        return "Welcome to LB Finance transfers! Please provide the payee name from your saved list and the amount you'd like to transfer."

def request_account_info(state: TransferState) -> str:
    """Request destination account using NOVA team approach with full user context."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user needs to specify a payee account. Ask them to specify the payee name from their saved list.
Be professional, concise (1 sentence), and mobile-friendly.

CRITICAL RULES:
- DO NOT list specific payee names in your response
- Just ask them to provide a payee name from their saved list

IMPORTANT: Return ONLY the question text that should be shown to the user. Do not include any explanations or analysis.

Example output: "Please specify which payee you'd like to transfer to from your saved list."
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM account request: {e}")
        return "Please specify the payee account name from your saved list."

def request_amount_info(state: TransferState) -> str:
    """Request transfer amount using NOVA team approach with full user context."""
    full_context = get_comprehensive_user_context(state)
    destination = state.get("destination_account", "")
    user_balance = state.get("user_balance", "$0.00")
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user needs to specify the transfer amount. They want to transfer to: {destination}
Ask for the amount without mentioning the balance.
Be professional, concise (1 sentence), and mobile-friendly.

CRITICAL RULES:
- ALWAYS mention the current destination account ({destination}) in your response
- DO NOT mention the user's current balance unless specifically asked
- Ask for the transfer amount

IMPORTANT: Return ONLY the question text that should be shown to the user. Do not include any explanations or analysis.

Example output: "How much would you like to transfer to {destination}?"
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM amount request: {e}")
        # Fallback that includes the destination account
        if destination:
            return f"How much would you like to transfer to {destination}?"
        else:
            return f"How much would you like to transfer?"

def generate_account_not_exist_message(account: str, state: TransferState) -> str:
    """Generate account not found message using LLM with full user context."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user tried to transfer to "{account}" but this payee is not in their saved payee list. 
Provide a helpful error message that:
1. Explains the payee is not found
2. Asks them to choose from their saved list
3. IMPORTANT: If the user asks to add a new payee, clearly explain that you CANNOT add new payees and they need to contact customer service
4. DO NOT list specific payee names unless they specifically ask

Be professional, concise, and helpful.

IMPORTANT: Return ONLY the message text that should be shown to the user. Do not include any explanations or analysis.

Example output: "The account '{account}' is not in your payee list. Please choose from your saved payees. I cannot add new payees - please contact customer service for assistance with adding new payees."
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM account not found message: {e}")
        return f"The specified payee account is not in your saved list. Please choose from your saved payees. I cannot add new payees - please contact customer service for assistance."

def generate_insufficient_balance_message(amount: str, state: TransferState) -> str:
    """Generate insufficient balance message using LLM with full user context."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user tried to transfer ${amount} but has insufficient funds. Provide a helpful error message that:
1. Explains insufficient funds
2. Only mention balance if absolutely necessary for user understanding
3. Asks for a lower amount

Be professional, empathetic, and helpful.

IMPORTANT: Return ONLY the message text that should be shown to the user. Do not include any explanations or analysis.

Example output: "Insufficient funds for ${amount} transfer. Please enter a lower amount."
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM insufficient balance message: {e}")
        return f"Insufficient funds for this transaction. Please enter a lower amount."

def generate_formal_error_response(error_message: str, account: str, amount: int, state: TransferState) -> str:
    """Generate a formal, user-friendly error response using LLM with full user context."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

A transfer operation has failed with this error: "{error_message}"
Account: {account}
Amount: ${amount}

Generate a professional, empathetic response that:
1. Acknowledges the issue without technical jargon
2. Explains what happened in simple terms  
3. Provides helpful next steps or alternatives
4. Maintains trust in LB Finance services

Be concise (1-2 sentences), formal but friendly, and mobile-optimized.

IMPORTANT: Return ONLY the message text that should be shown to the user. Do not include any explanations or analysis.

Examples:
- "We're unable to complete your transfer due to insufficient funds. Please check your balance and try again."
- "Your transfer couldn't be processed at this time. Please verify the recipient details and try again, or contact our support team."
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        # Fallback professional message if LLM fails
        return f"We're unable to complete your ${amount} transfer to {account} at this time. Please try again later or contact customer support."

def handle_payee_list_request(user_token: str, state: TransferState) -> str:
    """Handle requests to view the user's payee list with intelligent LLM response."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user is specifically asking about their payee list. This is the ONLY time you should show their actual payee names.
Based on their profile above, provide a helpful response that:
1. Shows their available payees (from the profile above) since they specifically asked
2. Asks which one they'd like to transfer to (if they have payees)
3. If they have no payees, explain that they need to contact customer service to add payees
4. IMPORTANT: If the user asks to add a new payee, clearly explain that you CANNOT add new payees and they need to contact customer service

Keep the response friendly, concise, and actionable.

IMPORTANT: Return ONLY the response text that should be shown to the user. Do not include any explanations or analysis.

Example outputs:
- "Here are your saved payees: Alice, Bob, Mike. Which one would you like to transfer to?"
- "You don't have any saved payees yet. Please contact customer service to add new payees to your account."
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM payee list request: {e}")
        payee_list = get_user_payees_list(user_token)
        if payee_list and payee_list != "No payees found":
            return f"Here are your saved payees: {payee_list}. Which one would you like to transfer to?"
        else:
            return "You don't have any saved payees yet. Please contact customer service to add new payees to your account."

def handle_balance_inquiry_request(user_token: str, state: TransferState) -> str:
    """Handle requests to check account balance with intelligent LLM response."""
    full_context = get_comprehensive_user_context(state)
    
    try:
        # Get balance directly from database (not using the tool which requires amount)
        from firebase import DatabaseManager
        db_manager = DatabaseManager("main_DB")
        balance = db_manager.get_account_balance(user_token)
        balance_result = f"Your current account balance is ${balance or '0.00'}"
        
        # Generate a friendly response with the balance
        prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user asked about their account balance. The balance check result is: "{balance_result}"

Generate a friendly, professional response that:
1. Shows their current balance clearly
2. Offers additional assistance if needed
3. Keeps it concise and friendly

IMPORTANT: Return ONLY the response text that should be shown to the user.

Example output: "Your current account balance is $1,800.00. Would you like to initiate a transaction or explore other services we offer?"
"""
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error in LLM balance response: {e}")
            # Fallback to direct balance result
            return balance_result
            
    except Exception as e:
        print(f"Error checking balance: {e}")
        return "I'm unable to retrieve your balance right now. Please try again later or contact customer support."

def handle_compound_request(user_token: str, state: TransferState) -> str:
    """Handle compound requests that ask for multiple things (e.g., payees + balance)."""
    full_context = get_comprehensive_user_context(state)
    
    # Get both payee list and balance information
    try:
        # Get payee list
        payee_list = get_user_payees_list(user_token)
        
        # Get balance directly from database (not using the tool which requires amount)
        try:
            from firebase import DatabaseManager
            db_manager = DatabaseManager("main_DB")
            balance = db_manager.get_account_balance(user_token)
            balance_result = f"Your current account balance is ${balance or '0.00'}"
        except Exception as e:
            print(f"Error getting balance from DB: {e}")
            balance_result = "Unable to retrieve balance at this time"
        
        # Generate a comprehensive response
        prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user asked a compound question requesting both payee information and balance. 

Available information:
- Payee list: {payee_list}
- Balance check result: "{balance_result}"

Generate a comprehensive response that:
1. Shows their saved payees (since they asked for this)
2. Shows their current balance (since they asked for this)
3. Offers additional assistance
4. Keeps it organized and friendly

IMPORTANT: Return ONLY the response text that should be shown to the user.

Example output: "Here are your saved payees: Sarah, Mike, Emma. Your current account balance is $1,800.00. Would you like to initiate a transaction or explore other services?"
"""
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error in LLM compound response: {e}")
            # Fallback to simple combination
            if payee_list and payee_list != "No payees found":
                return f"Here are your saved payees: {payee_list}. {balance_result}. Would you like to initiate a transaction?"
            else:
                return f"You don't have any saved payees yet. {balance_result}. Please contact customer service to add new payees to your account."
            
    except Exception as e:
        print(f"Error handling compound request: {e}")
        return "I'm having trouble retrieving your information right now. Please try asking about payees and balance separately."

def handle_add_payee_request(user_token: str, state: TransferState) -> str:
    """Handle requests to add new payees - explain that the agent cannot do this."""
    full_context = get_comprehensive_user_context(state)
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user is asking to add a new payee to their account. You CANNOT add new payees through this chat interface.

Provide a helpful response that:
1. Clearly explains that you cannot add new payees
2. Directs them to contact customer service or use other banking channels
3. Offers to help with other banking services you CAN provide

Keep the response friendly, professional, and helpful.

IMPORTANT: Return ONLY the response text that should be shown to the user. Do not include any explanations or analysis.

Example output: "I'm unable to add new payees through this chat interface. Please contact our customer service team or visit a branch to add new payees to your account. I can help you with transfers to existing payees, balance inquiries, and other banking services."
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM add payee response: {e}")
        return "I'm unable to add new payees through this chat interface. Please contact our customer service team or visit a branch to add new payees to your account. I can help you with transfers to existing payees, balance inquiries, and other banking services."

def classify_user_confirmation(user_input: str, state: TransferState) -> dict:
    """Use LLM to classify whether user is confirming or declining the transfer."""
    print(f"🤔 CONFIRMATION ANALYSIS: {user_input}")
    
    # Get context for the confirmation
    destination = state.get("destination_account", "Unknown")
    amount = state.get("transfer_amount", "Unknown")
    
    prompt = f"""
You are a specialized AI assistant that classifies user responses to transfer confirmation requests.

CONTEXT:
The user was asked to confirm a transfer of ${amount} to {destination}.

USER RESPONSE: "{user_input}"

Your task is to determine if the user is:
1. CONFIRMING the transfer (positive response)
2. DECLINING the transfer (negative response)
3. UNCLEAR (ambiguous or unrelated response)

CONFIRMATION INDICATORS (POSITIVE):
- "yes", "ok", "okay", "sure", "confirm", "proceed", "go ahead"
- "that's correct", "that's right", "correct"
- "do it", "send it", "transfer it"
- "continue", "yup", "yeah", "yep"
- Variations like "yessssss", "syre" (typo for sure), "okkk", "yesss"
- Any clear positive affirmation including typos and repeated letters

DECLINE INDICATORS (NEGATIVE):
- "no", "not now", "cancel", "stop", "don't", "abort"
- "that's wrong", "incorrect", "mistake"
- "wait", "hold on", "let me think"
- "nope", "nah", "never mind"
- Any clear negative response

UNCLEAR INDICATORS:
- Questions about something else
- Requests to change amount/recipient (e.g., "transfer 100 to Paul", "send 200", "to Alice")
- Providing new transaction details rather than confirming existing ones
- Unrelated responses
- Ambiguous statements

IMPORTANT: 
- Be flexible with typos and repeated letters (e.g., "yessssss" = yes, "syre" = sure)
- If the user provides specific transaction details (amounts, payee names, "transfer X to Y"), classify as "unclear"

Return ONLY a JSON object with this exact format:
{{
  "classification": "confirm" | "decline" | "unclear",
  "confidence": <0-100>,
  "reasoning": "brief explanation"
}}

Examples:
- "yes" → {{"classification": "confirm", "confidence": 95, "reasoning": "Clear positive affirmation"}}
- "yessssss" → {{"classification": "confirm", "confidence": 90, "reasoning": "Positive affirmation with repeated letters"}}
- "syre" → {{"classification": "confirm", "confidence": 85, "reasoning": "Typo for 'sure' - positive affirmation"}}
- "continue" → {{"classification": "confirm", "confidence": 90, "reasoning": "Request to proceed"}}
- "no" → {{"classification": "decline", "confidence": 95, "reasoning": "Clear negative response"}}
- "what about my balance?" → {{"classification": "unclear", "confidence": 80, "reasoning": "Question about different topic"}}
- "transfer 100 to Paul" → {{"classification": "unclear", "confidence": 90, "reasoning": "Providing new transaction details rather than confirming"}}
- "send 200" → {{"classification": "unclear", "confidence": 85, "reasoning": "Providing new amount rather than confirming"}}
"""
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=user_input)]
    
    try:
        response = llm.invoke(messages).content.strip()
        print(f"🤔 Confirmation LLM response: {response}")
        parsed = json.loads(response)
        
        if isinstance(parsed, dict):
            # Ensure all required fields exist
            parsed.setdefault("classification", "unclear")
            parsed.setdefault("confidence", 50)
            parsed.setdefault("reasoning", "Unable to classify")
            return parsed
            
    except Exception as e:
        print(f"🤔 Confirmation classification error: {e}")
        
    # Enhanced fallback with better pattern matching for confirmations
    user_input_lower = user_input.strip().lower()
    
    # Positive confirmation patterns (including typos and variations)
    positive_patterns = [
        "yes", "yep", "yeah", "yup", "ok", "okay", "sure", "confirm", "proceed", "go ahead",
        "correct", "right", "do it", "send it", "transfer it", "continue", "good", "fine"
    ]
    
    # Check for positive patterns with typos and repeated letters
    for pattern in positive_patterns:
        # Check exact match
        if user_input_lower == pattern:
            return {"classification": "confirm", "confidence": 90, "reasoning": f"Exact match for '{pattern}'"}
        
        # Check for repeated letters (e.g., "yessssss")
        if pattern in user_input_lower and len(user_input_lower) > len(pattern):
            # Allow for repeated characters
            if user_input_lower.replace(pattern[-1], '').strip() == pattern[:-1]:
                return {"classification": "confirm", "confidence": 85, "reasoning": f"Variation of '{pattern}' with repeated letters"}
    
    # Check for common typos
    typo_mappings = {
        "syre": "sure",
        "ys": "yes", 
        "ye": "yes",
        "ya": "yes",
        "okkk": "ok",
        "oka": "ok",
        "contineu": "continue",
        "continu": "continue"
    }
    
    for typo, correct in typo_mappings.items():
        if user_input_lower == typo:
            return {"classification": "confirm", "confidence": 80, "reasoning": f"Typo for '{correct}' - positive confirmation"}
    
    # Negative confirmation patterns
    negative_patterns = ["no", "nope", "nah", "cancel", "stop", "abort", "wait", "hold", "incorrect", "wrong", "mistake"]
    
    for pattern in negative_patterns:
        if pattern in user_input_lower:
            return {"classification": "decline", "confidence": 85, "reasoning": f"Contains negative indicator '{pattern}'"}
    
    # If it contains specific transaction details, it's unclear
    if any(word in user_input_lower for word in ["transfer", "send", "to ", "$", "alice", "bob", "mike", "paul"]):
        return {"classification": "unclear", "confidence": 75, "reasoning": "Contains new transaction details rather than confirmation"}
    
    # Fallback: return safe default
    return {
        "classification": "unclear",
        "confidence": 30,
        "reasoning": "Unable to classify - defaulting to unclear"
    }

def generate_confirmation_message(state: TransferState) -> str:
    """Generate transfer confirmation message using LLM."""
    full_context = get_comprehensive_user_context(state)
    destination = state.get("destination_account", "Unknown")
    amount = state.get("transfer_amount", "Unknown")
    
    prompt = f"""
You are LEO, LB Finance's intelligent virtual assistant.

{full_context}

The user has provided all transfer details. Generate a clear confirmation message that:
1. Shows the transfer details (amount and recipient)
2. Asks for explicit confirmation
3. Keeps it professional and concise (1-2 sentences)

Transfer Details:
- Amount: ${amount}
- Recipient: {destination}

IMPORTANT: Return ONLY the confirmation message text that should be shown to the user. Do not include any explanations or analysis.

Example output: "Do you confirm to transfer $3000 to Mike? Please confirm to proceed."
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in LLM confirmation message: {e}")
        return f"Do you confirm to transfer ${amount} to {destination}? Please confirm to proceed."

# ───────────────────────────────────────────────────────────────
# GRAPH NODES
# ───────────────────────────────────────────────────────────────

def process_user_input(state: TransferState) -> TransferState:
    """Process user input using NOVA team approach with conversation memory."""
    try:
        user_query = state.get("user_query", "")
        user_token = state.get("user_token", "")

        print(f"🔵 PROCESS_INPUT: {user_query}")

        # Check if we're waiting for confirmation
        if state.get("confirmation_requested"):
            print("🤔 Processing confirmation response")
            confirmation_result = classify_user_confirmation(user_query, state)
            
            if confirmation_result["classification"] == "confirm":
                print("✅ User confirmed transfer")
                state["user_confirmed"] = True
                state["confirmation_requested"] = False
                state["ai_response"] = "Thank you for confirming. Processing your transfer..."
                # Add AI response to conversation history
                state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
                return state
            elif confirmation_result["classification"] == "decline":
                print("❌ User declined transfer")
                state["user_confirmed"] = False
                state["confirmation_requested"] = False
                state["transaction_result"] = "CANCELLED"
                state["ai_response"] = "Transfer cancelled. How else can I help you today?"
                # Clear transaction data
                state["destination_account"] = ""
                state["transfer_amount"] = ""
                # Add AI response to conversation history
                state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
                return state
            else:
                print("❓ Unclear confirmation response - treating as new input")
                # Reset confirmation flags and reprocess as new input
                state["confirmation_requested"] = False
                state["user_confirmed"] = False
                # Don't return here - let it continue to normal processing

        # Check if previous transaction was completed and clear it for new transaction
        if state.get("transaction_result") in ["COMPLETED", "ERROR", "CANCELLED"]:
            print(f"🔄 Clearing completed transaction state (status: {state.get('transaction_result')}) for new input")
            state["destination_account"] = ""
            state["transfer_amount"] = ""
            state["transaction_result"] = ""
            state["conversation_context"] = ""
            state["needs_confirmation"] = False
            state["confirmation_requested"] = False
            state["user_confirmed"] = False

        # Add user message to conversation history
        state = memory_server.add_message_to_history(state, "user", user_query)
        print(f"🔍 After adding user message to history: chat_history length = {len(state.get('chat_history', []))}")

        # Load user profile if not already loaded
        state = load_user_profile(user_token, state)

        # Get conversation context for NOVA analysis
        conversation_context = memory_server.get_conversation_context(state.get('thread_id', ''), state)
        print(f"🧠 Conversation Context: {conversation_context[:100]}...")

        # Check if we're in continuation mode (has existing transaction data)
        is_transaction_continuation = bool(
            state.get("destination_account") or 
            state.get("transfer_amount")
        )

        # Use NOVA team to analyze intent and extract data with conversation context
        nova_analysis = analyze_user_intent_and_extract_data(user_query, user_token, state)
        intent = nova_analysis.get("intent", "general")
        confidence = nova_analysis.get("confidence", 50)
        extracted_account = nova_analysis.get("account", "None")
        extracted_amount = nova_analysis.get("amount", None)
        
        print(f"🧠 NOVA Analysis Details:")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence}%")
        print(f"   Extracted Account: {extracted_account}")
        print(f"   Extracted Amount: {extracted_amount}")
        print(f"   Transaction Continuation: {is_transaction_continuation}")
        print(f"🧠 NOVA Raw Analysis: {nova_analysis}")  # Added debug line

        # Handle general inquiries (redirect to banking functions regardless of transaction state)
        if intent == "general":
            print("🔵 Handling general inquiry via NOVA")
            ai_response = handle_general_inquiry(user_query, user_token, state)
            state["ai_response"] = ai_response
            # Add AI response to conversation history
            state = memory_server.add_message_to_history(state, "assistant", ai_response)
            return state

        # Handle payee list requests
        if intent == "payee_list":
            print("🔵 Handling payee list request")
            ai_response = handle_payee_list_request(user_token, state)
            state["ai_response"] = ai_response
            # Add AI response to conversation history
            state = memory_server.add_message_to_history(state, "assistant", ai_response)
            return state

        # Handle add payee requests
        if intent == "add_payee":
            print("🔵 Handling add payee request")
            ai_response = handle_add_payee_request(user_token, state)
            state["ai_response"] = ai_response
            # Add AI response to conversation history
            state = memory_server.add_message_to_history(state, "assistant", ai_response)
            return state

        # Handle balance inquiry requests
        if intent == "balance_inquiry":
            print("🔵 Handling balance inquiry request")
            ai_response = handle_balance_inquiry_request(user_token, state)
            state["ai_response"] = ai_response
            # Add AI response to conversation history
            state = memory_server.add_message_to_history(state, "assistant", ai_response)
            return state

        # Handle compound requests (multiple questions)
        if intent == "compound":
            print("🔵 Handling compound request")
            ai_response = handle_compound_request(user_token, state)
            state["ai_response"] = ai_response
            # Add AI response to conversation history
            state = memory_server.add_message_to_history(state, "assistant", ai_response)
            return state

        # Transaction intent detected or continuation mode
        print("🔵 Processing transaction intent")

        # Extract account and amount from NOVA analysis - be more explicit
        print(f"🔍 State before extraction:")
        print(f"   Current destination_account: '{state.get('destination_account')}'")
        print(f"   Current transfer_amount: '{state.get('transfer_amount')}'")
        
        # First, extract account if available - let LLM handle correction detection
        current_account = state.get("destination_account")
        should_update_account = False
        
        # Check if we should update the account
        if extracted_account and extracted_account != "None":
            if not current_account or current_account in [None, "None", ""]:
                # No current account - set new one
                should_update_account = True
                print(f"📍 Setting new destination account: {extracted_account}")
            elif current_account != extracted_account:
                # Account changed - LLM detected this should be updated
                should_update_account = True
                print(f"🔄 LLM detected account change: '{current_account}' → '{extracted_account}'")
        else:
            print(f"🔍 Account update check: extracted_account='{extracted_account}', current_account='{current_account}'")
            print(f"🔍 Account update condition: extracted_account and extracted_account != 'None' = {extracted_account and extracted_account != 'None'}")
        
        if should_update_account:
            state["destination_account"] = extracted_account
            print(f"✅ NOVA extracted/updated account: {extracted_account}")
            
            # Update conversation context after account change
            state = memory_server.update_conversation_context(state)
            
            # IMMEDIATE PAYEE VALIDATION - Check if this account exists in payee list
            try:
                from tools.checkAccountExistence import CheckAccountExistence
                from toolInputs.checkAccountExistenceInput import CheckAccountExistenceInput
                
                check_existence_tool = CheckAccountExistence()
                input_data = CheckAccountExistenceInput(account=extracted_account, user_token=user_token)
                
                print(f"🔍 Immediately validating payee: {extracted_account}")
                account_check_result = check_existence_tool.check_existence(input_data)
                
                # If the payee doesn't exist, clear the account and respond immediately
                if "not in your payees list" in account_check_result.lower() or "not found" in account_check_result.lower():
                    print(f"❌ Payee '{extracted_account}' not found - clearing and asking for valid payee")
                    state["destination_account"] = None  # Clear invalid account
                    state["ai_response"] = generate_account_not_exist_message(extracted_account, state)
                    # Add AI response to conversation history
                    state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
                    return state
                else:
                    print(f"✅ Payee '{extracted_account}' validated successfully")
            except Exception as e:
                print(f"⚠️ Error validating payee immediately: {e}")
                # On error, clear the account and ask for valid payee
                state["destination_account"] = None
                state["ai_response"] = generate_account_not_exist_message(extracted_account, state)
                state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
                return state
        else:
            print(f"⚠️ Account extraction skipped. Current: '{current_account}', Extracted: '{extracted_account}'")

        # Then extract amount if available - let LLM handle correction detection
        current_amount = state.get("transfer_amount")
        should_update_amount = False
        
        # Check if we should update the amount
        if extracted_amount is not None:
            if not current_amount or current_amount in [None, "None", ""]:
                # No current amount - set new one
                should_update_amount = True
                print(f"💰 Setting new transfer amount: ${extracted_amount}")
            elif str(current_amount).strip() != str(extracted_amount).strip():
                # Amount changed - LLM detected this should be updated
                should_update_amount = True
                print(f"🔄 LLM detected amount change: '${current_amount}' → '${extracted_amount}'")
        
        # Special case: if user just provides a number and we have a destination account, treat it as amount
        if (not should_update_amount and 
            extracted_amount is not None and 
            state.get("destination_account") and 
            not state.get("transfer_amount")):
            should_update_amount = True
            print(f"💰 Setting amount for existing destination: ${extracted_amount}")
        
        if should_update_amount:
            # Set the extracted amount directly - balance validation will be done later during transaction
            state["transfer_amount"] = str(extracted_amount)
            print(f"✅ Amount updated to: ${extracted_amount}")
            state = memory_server.update_conversation_context(state)
            
            print(f"✅ NOVA extracted/updated amount: ${extracted_amount}")
        else:
            print(f"⚠️ Amount extraction skipped. Current: '${current_amount}', Extracted: '{extracted_amount}'")

        # Determine what to ask for based on current state
        has_destination = state.get("destination_account") and state.get("destination_account") != "None"
        has_amount = state.get("transfer_amount") and state.get("transfer_amount") not in ["None", ""]

        print(f"🔍 Current state check: Dest={has_destination}, Amount={has_amount}")

        if has_destination and has_amount:
            # Both account and amount are available, request confirmation
            print("✅ Both account and amount available, requesting confirmation...")
            state["needs_confirmation"] = True
            state["confirmation_requested"] = True
            state["ai_response"] = generate_confirmation_message(state)
        elif not has_destination and not has_amount:
            state["ai_response"] = generate_welcome_message(state)
        elif not has_destination:
            state["ai_response"] = request_account_info(state)
        elif not has_amount:
            state["ai_response"] = request_amount_info(state)
        else:
            # This should not happen, but just in case
            state["ai_response"] = "Processing your transfer request..."

        # Add AI response to conversation history if we have one
        if state.get("ai_response"):
            state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
        
        return state
        
    except Exception as e:
        print(f"❌ Error in process_user_input: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure we always return a valid state with an error response
        state["ai_response"] = "I understand you want to use our banking services. I can help you with money transfers, checking your account balance, and managing your payee list. How can I assist you today?"
        # Add AI response to conversation history
        try:
            state = memory_server.add_message_to_history(state, "assistant", state["ai_response"])
        except:
            pass  # Don't fail if we can't add to history
        return state

def validate_transfer_data(state: TransferState) -> TransferState:
    """Validate that all required transfer data is present and account exists."""
    print("🔍 VALIDATE_DATA")

    # Skip validation for general inquiry responses
    current_response = state.get("ai_response", "")
    if ("lb finance" in current_response.lower() and 
        not ("transfer" in current_response.lower() or "send" in current_response.lower())):
        print("🔍 Skipping validation for general inquiry")
        return state

    # Check required fields
    destination_account = state.get("destination_account")
    amount = state.get("transfer_amount")
    user_token = state.get("user_token")

    if not all([destination_account, amount, user_token]) or \
       any(val in [None, "None", ""] for val in [destination_account, amount]):
        print("🔍 Missing required fields - no validation needed yet")
        return state

    # Validate account existence using the tool
    try:
        from tools.checkAccountExistence import CheckAccountExistence
        from toolInputs.checkAccountExistenceInput import CheckAccountExistenceInput
        
        check_existence_tool = CheckAccountExistence()
        input_data = CheckAccountExistenceInput(account=destination_account, user_token=user_token)
        
        print(f"🔍 Checking account existence: {destination_account}")
        account_check_result = check_existence_tool.check_existence(input_data)
        
        # If the result indicates account doesn't exist
        if "not in your payees list" in account_check_result.lower() or "not found" in account_check_result.lower():
            print("🔍 Account does not exist")
            state["ai_response"] = generate_account_not_exist_message(destination_account, state)
            state["destination_account"] = None
            return state
    except Exception as e:
        print(f"🔍 Error checking account existence: {e}")
        state["ai_response"] = generate_account_not_exist_message(destination_account, state)
        state["destination_account"] = None
        return state

    # Validate balance using the tool
    try:
        from tools.checkAccountBalance import CheckAccountBalance
        from toolInputs.checkAccountBalanceInput import CheckAccountBalanceInput
        
        check_balance_tool = CheckAccountBalance()
        
        amount_value = float(amount.replace(",", "").replace("$", "").strip())
        input_data = CheckAccountBalanceInput(amount=int(amount_value), user_token=user_token)
        
        print(f"🔍 Checking balance for amount: {amount_value}")
        balance_check_result = check_balance_tool.check_balance(input_data)
        
        if "Insufficient balance" in balance_check_result or "insufficient" in balance_check_result.lower():
            print("🔍 Insufficient balance")
            state["ai_response"] = generate_insufficient_balance_message(amount, state)
            state["transfer_amount"] = None
            return state
        
        print("🔍 Validation passed")
        
        # Check if we need confirmation instead of just saying data is valid
        if not state.get("confirmation_requested") and not state.get("user_confirmed"):
            print("🔍 All data valid, requesting confirmation...")
            state["needs_confirmation"] = True
            state["confirmation_requested"] = True
            state["ai_response"] = generate_confirmation_message(state)
        else:
            # If confirmation was already requested, keep the existing response
            print("🔍 Validation passed, confirmation already requested")
        
    except ValueError:
        state["ai_response"] = "Invalid amount format. Please specify the amount correctly."
        state["transfer_amount"] = None
    except Exception as e:
        print(f"🔍 Error checking balance: {e}")
        state["ai_response"] = "Unable to verify balance. Please try again."
        state["transfer_amount"] = None

    return state

def complete_transfer(state: TransferState) -> TransferState:
    """Complete the transfer process using the banking API."""
    print("💸 COMPLETE_TRANSFER")
    
    source_account = state.get("user_account")  # Use user_account as source
    destination_account = state.get("destination_account")
    amount = state.get("transfer_amount")
    user_token = state.get("user_token")

    print(f"💸 Transfer details: {amount} from {source_account} to {destination_account}")

    try:
        amount_int = int(amount.replace(",", "").replace("$", "").strip())
    except ValueError:
        state["ai_response"] = "Invalid amount format. Please specify the amount correctly."
        return state

    # Use the ProcessTransfer tool directly
    try:
        from toolInputs.processTransferInput import ProcessTransferInput
        from tools.processTransfer import ProcessTransfer
        
        input_data = ProcessTransferInput(
            source_account=source_account,
            destination_account=destination_account,
            amount=amount_int,
            user_token=user_token
        )
        
        tool = ProcessTransfer()
        response = tool.process_transfer(input_data)
        print(f"💸 Transfer response: {response}")
        
        # Set transaction_result based on success/failure for frontend display
        if response.startswith("Transfer successful"):
            state["transaction_result"] = "COMPLETED"
            state["ai_response"] = f"✅ Success! Transferred ${amount_int} to {destination_account}."
            
            # Refresh user balance after successful transfer
            try:
                from firebase import DatabaseManager
                db_manager = DatabaseManager("main_DB")
                updated_balance = db_manager.get_account_balance(user_token)
                
                # Update the state with new balance
                state["user_balance"] = f"${updated_balance}" if updated_balance else "$0.00"
                print(f"💰 Updated balance in state: {state['user_balance']}")
                
            except Exception as e:
                print(f"⚠️ Error refreshing balance after transfer: {e}")
                # Don't fail the transfer if balance refresh fails
                
        else:
            state["transaction_result"] = "ERROR"
            # Generate formal error response using LLM
            error_response = generate_formal_error_response(response, destination_account, amount_int, state)
            state["ai_response"] = error_response
            
    except Exception as e:
        # Generate formal error response for technical errors
        error_response = generate_formal_error_response(f"Technical error: {str(e)}", destination_account if 'destination_account' in locals() else "Unknown", amount_int if 'amount_int' in locals() else 0, state)
        state["ai_response"] = error_response
        state["transaction_result"] = "ERROR"

    return state

def reset_state_after_success(state: TransferState) -> TransferState:
    """Reset state after transaction for new transfers, but preserve completed transaction info for display."""
    print("🔄 RESET_STATE")
    
    # The transaction_result status has already been set correctly in complete_transfer
    current_status = state.get("transaction_result", "ERROR")
    print(f"🔄 Current transaction status: {current_status}")
    
    # Keep the completed transaction information for display purposes
    # This will be cleared on the next user input
    reset_state = TransferState(
        # User Profile Information (keep these)
        user_name=state.get("user_name", ""),
        user_account=state.get("user_account", ""),
        user_balance=state.get("user_balance", ""),
        user_status=state.get("user_status", ""),
        
        # Transaction Flow Data (keep completed transaction for display)
        user_query="",
        destination_account=state.get("destination_account", ""),  # Keep for display
        transfer_amount=state.get("transfer_amount", ""),          # Keep for display
        user_token=state.get("user_token", ""),
        ai_response=state.get("ai_response", ""),
        transaction_result=current_status,  # Preserve the status set in complete_transfer
        
        # Confirmation Management (reset for new transactions)
        needs_confirmation=False,
        confirmation_requested=False,
        user_confirmed=False,
        
        # Memory & Conversation Management (keep conversation history)
        chat_history=state.get("chat_history", []),
        conversation_context="",  # Reset transaction context for new transactions
        turn_number=state.get("turn_number", 0),
        thread_id=state.get("thread_id", ""),
        
        # State Management Flags (keep profile data)
        profile_loaded=state.get("profile_loaded", False),
        last_balance_check=state.get("last_balance_check", "")
    )
    
    return reset_state

def finalize_process(state: TransferState) -> TransferState:
    """Final cleanup and state preparation."""
    print("🏁 FINALIZE")
    
    return state

# ───────────────────────────────────────────────────────────────
# GRAPH CONDITIONS
# ───────────────────────────────────────────────────────────────

def has_complete_transfer_data(state: TransferState) -> str:
    """Check if all required transfer data is present and confirmed."""
    destination_account = state.get("destination_account")
    amount = state.get("transfer_amount")
    user_token = state.get("user_token")
    user_confirmed = state.get("user_confirmed", False)

    has_destination = destination_account and destination_account != "None"
    has_amount = amount and amount not in ["None", "", "Insufficient balance"]
    has_token = user_token and user_token != ""

    is_complete = has_destination and has_amount and has_token and user_confirmed

    print(f"🔍 Data completeness check: {is_complete}")
    print(f"   Dest: {has_destination}, Amount: {has_amount}, Token: {has_token}, Confirmed: {user_confirmed}")

    return "complete" if is_complete else "incomplete"

# ───────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ───────────────────────────────────────────────────────────────

def create_banking_graph():
    """Create and return the banking workflow graph."""
    
    builder = StateGraph(TransferState)
    
    # Add nodes
    builder.add_node("process_input", process_user_input)
    builder.add_node("validate_data", validate_transfer_data)
    builder.add_node("complete_transfer", complete_transfer)
    builder.add_node("reset_state", reset_state_after_success)
    builder.add_node("finalize", finalize_process)

    # Add edges
    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "validate_data")
    
    builder.add_conditional_edges(
        "validate_data",
        has_complete_transfer_data,
        {
            "complete": "complete_transfer",
            "incomplete": "finalize"
        }
    )
    
    builder.add_edge("complete_transfer", "reset_state")
    builder.add_edge("reset_state", "finalize")
    builder.add_edge("finalize", END)

    # Create memory for state persistence
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    return graph

# ───────────────────────────────────────────────────────────────
# MAIN EXECUTION FUNCTION
# ───────────────────────────────────────────────────────────────────────────────

def execute_banking_graph(thread_id: str, user_input: str, user_token: str, current_state: dict = None) -> dict:
    """Execute the banking graph and return structured response."""
    
    print(f"🚀 EXECUTE_GRAPH: {user_input} (User: {user_token})")
    
    # Basic validation
    if not user_input or not user_token:
        print("❌ Missing required parameters")
        return {
            "message": "Invalid request parameters",
            "response_type": "error", 
            "status": "error",
            "transfer_state": current_state or {}
        }
    
    try:
        # Create graph
        print("🔧 Creating graph...")
        graph = create_banking_graph()
        print("✅ Graph created successfully")
        
        # Setup thread config
        thread_config = {"configurable": {"thread_id": thread_id}}

        # Use provided current state or create fresh state with memory initialization
        if current_state:
            print(f"🔄 Using provided conversation state - Dest: '{current_state.get('destination_account', '')}', Amount: '{current_state.get('transfer_amount', '')}'")
            initial_state = current_state.copy()
            initial_state["user_query"] = user_input  # Update with new user input
            initial_state["thread_id"] = thread_id  # Ensure thread_id is set
            
            # Initialize memory fields if not present
            if 'chat_history' not in initial_state:
                initial_state['chat_history'] = []
            if 'conversation_context' not in initial_state:
                initial_state['conversation_context'] = ""
            if 'turn_number' not in initial_state:
                initial_state['turn_number'] = 0
        else:
            print("🆕 Starting with fresh state")
            # Prepare fresh initial state with memory initialization
            initial_state = TransferState(
                # User Profile Information
                user_name="",
                user_account="",
                user_balance="",
                user_status="",
                # Transaction Flow Data
                user_query=user_input,
                destination_account="", 
                transfer_amount="",
                user_token=user_token,
                ai_response="",
                transaction_result="",
                # Confirmation Management
                needs_confirmation=False,
                confirmation_requested=False,
                user_confirmed=False,
                # Memory & Conversation Management
                chat_history=[],
                conversation_context="",
                turn_number=0,
                thread_id=thread_id,
                # State Management Flags
                profile_loaded=False,
                last_balance_check=""
            )

        # Update conversation context
        print("🔧 Updating conversation context...")
        initial_state = memory_server.update_conversation_context(initial_state)
        print("✅ Conversation context updated")

        print(f"🔵 Initial state prepared with user_query: {initial_state.get('user_query', 'N/A')}")

        # Execute graph with better error handling
        print("🔧 Executing graph...")
        final_state = None
        step_count = 0
        
        for step_count, update in enumerate(graph.stream(initial_state, thread_config), start=1):
            print(f"🔄 Step {step_count}: {list(update.keys())}")
            final_state = list(update.values())[0] if update else final_state
            
            # Check if we have a valid response
            if final_state and final_state.get("ai_response"):
                print(f"✅ Got AI response in step {step_count}: {final_state.get('ai_response')[:100]}...")
            
            if step_count > 10:  # Safety limit
                print("⚠️ Safety limit reached")
                break
        
        print(f"✅ Graph execution completed in {step_count} steps")
                
    except Exception as e:
        print(f"❌ Graph execution error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "message": f"Sorry, an error occurred: {str(e)}",
            "response_type": "error",
            "status": "error",
            "transfer_state": current_state or {}
        }

    # Get final state
    if not final_state:
        print("⚠️ No final state from graph execution, trying to get state...")
        try:
            final_state = graph.get_state(thread_config).values
            print("✅ Retrieved state from graph")
        except Exception as e:
            print(f"⚠️ Could not retrieve state: {e}")
            final_state = initial_state

    print(f"🏁 Final state: ai_response = '{final_state.get('ai_response', 'No response')[:100]}...'")

    # Build response
    ai_response = final_state.get("ai_response", "No response generated")
    if not ai_response or ai_response == "No response generated":
        print("⚠️ No AI response generated, using fallback")
        ai_response = "I understand you want to use our banking services. I can help you with money transfers, checking your account balance, and managing your payee list. How can I assist you today?"
    
    response = {
        "message": ai_response,
        "transfer_state": final_state,
        "user_token": user_token
    }

    # Determine response type and status
    if final_state.get("transaction_result"):
        response["response_type"] = "transaction"
        # Check for both successful completion and COMPLETED status
        if ("successful" in final_state.get("transaction_result", "").lower() or 
            final_state.get("transaction_result") == "COMPLETED"):
            response["status"] = "success"
        else:
            response["status"] = "error"
    elif any(final_state.get(field) for field in ["destination_account", "transfer_amount"]):
        response["response_type"] = "transaction"
        response["status"] = "pending"
        response["needs_more_info"] = True
    else:
        response["response_type"] = "general"
        # Don't set status for general conversations - status should only apply to transactions

    print(f"🏁 Final response: {response}")
    return response