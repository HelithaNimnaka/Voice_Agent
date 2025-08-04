import streamlit as st
import sys
import os
from typing_extensions import TypedDict
import sounddevice as sd
import soundfile as sf
import tempfile

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from graph_agent import execute_banking_graph, TransferState
    from firebase import DatabaseManager
    from functions import (
        transcribe, transcribe_uploaded_file, transcribe_audio_data,
        record_audio_smart, record_and_transcribe, record_and_transcribe_pure_memory,
        load_audio_any
    )
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

def is_admin():
    """Check if running with administrator privileges on Windows."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_audio_devices():
    """Check available audio input devices and return device info."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            return None, "No audio input devices found"
        
        # Get default input device
        try:
            default_device = sd.default.device[0]  # Input device index
            default_device_info = sd.query_devices(default_device, 'input')
            return default_device, f"Using device: {default_device_info['name']}"
        except Exception as e:
            # Use first available input device
            first_device = input_devices[0]
            return devices.tolist().index(first_device), f"Using device: {first_device['name']}"
            
    except Exception as e:
        return None, f"Error checking audio devices: {e}"

def record_audio_safely(duration, samplerate=16000, channels=1):
    """Safely record audio with comprehensive error handling for Bluetooth headsets."""
    try:
        # Check audio devices first
        device_id, device_message = check_audio_devices()
        
        # Multiple recording strategies for better compatibility
        strategies = []
        
        # Strategy 1: Use detected device if available
        if device_id is not None:
            strategies.append(("detected_device", device_id))
        
        # Strategy 2: Try default input device
        strategies.append(("default", None))
        
        # Strategy 3: Find alternative input devices
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if (device['max_input_channels'] > 0 and 
                    i != device_id and 
                    'Bluetooth' not in device['name']):  # Prefer non-Bluetooth for stability
                    strategies.append(("alternative", i))
                    break
        except:
            pass
        
        # Try different sample rates for each strategy
        sample_rates = [samplerate, 44100, 22050, 48000] if samplerate != 44100 else [44100, 16000, 22050, 48000]
        
        last_error = None
        
        for strategy_name, device in strategies:
            for rate in sample_rates:
                try:
                    st.info(f"🎤 Trying {strategy_name} at {rate}Hz...")
                    
                    if device is not None:
                        rec = sd.rec(
                            int(duration * rate), 
                            samplerate=rate, 
                            channels=channels,
                            device=device,
                            dtype='float32'  # Explicit dtype for compatibility
                        )
                    else:
                        rec = sd.rec(
                            int(duration * rate), 
                            samplerate=rate, 
                            channels=channels,
                            dtype='float32'
                        )
                    
                    sd.wait()
                    st.success(f"✅ Recording successful with {strategy_name} at {rate}Hz")
                    
                    # Resample to target rate if needed
                    if rate != samplerate:
                        import scipy.signal
                        rec = scipy.signal.resample(rec, int(len(rec) * samplerate / rate))
                    
                    return rec, None
                    
                except Exception as e:
                    last_error = e
                    st.warning(f"❌ {strategy_name} at {rate}Hz failed: {str(e)}")
                    continue
        
        # If all strategies fail
        return None, f"Unable to record audio. Last error: {str(last_error)}. Please check microphone permissions or try a different device."
        
    except Exception as e:
        return None, f"Audio recording error: {str(e)}"

def display_user_profile(transfer_state: TransferState):
    """Display user profile in clean format with debug details always available."""
    
    # Get user token from transfer state
    user_token = transfer_state.get('user_token', st.session_state.user_token)
    
    # Get actual user details from database
    user_details = get_user_details(user_token)
    
    # Use database details for display - no fallback values
    user_name = user_details.get('name', 'No Name Available')
    user_account = user_details.get('account', 'No Account Available')
    user_status = user_details.get('status', 'Unknown Status')
    
    # Prioritize balance from transfer state if available, otherwise use database balance
    if transfer_state.get('user_balance'):
        user_balance = transfer_state.get('user_balance')
    else:
        user_balance = user_details.get('balance', 'No Balance Available')
    
    # Update current balance if there was a successful transfer
    if (transfer_state.get('transaction_result') in ["SUCCESS", "COMPLETED"] and 
        transfer_state.get('transfer_amount')):
        user_balance = get_current_balance(user_token)
    
    # Get payees list
    from graph_agent import get_user_payees_list
    payees_list = get_user_payees_list(user_token)
    
    # Clean Frontend Display
    status_color = "🟢" if user_status == "Active" else "🔴"
    st.markdown(f"""
    👤 **Current User Details:**
    
    👤 **Name:** {user_name}  
    🏦 **Account:** {user_account}  
    💰 **Balance:** {user_balance}  
    {status_color} **Status:** {user_status}  
    👥 **My Payees:** {payees_list}
    """)
    
    # Debug Details (always available)
    with st.expander("🐛 Debug - Full State Information", expanded=False):
        st.markdown("**Complete Transfer State:**")
        for key, value in transfer_state.items():
            st.write(f"**{key}:** {value}")

# ─── Database Functions ──────────────────────────────────────────────

def update_transfer_state(user_query: str = None, ai_response: str = None, 
                         destination_account: str = None,
                         transfer_amount: str = None, redirect_url: str = None,
                         transaction_result: str = None, user_balance: str = None):
    """Update specific fields in the transfer state"""
    if user_query is not None:
        st.session_state.transfer_state['user_query'] = user_query
    if ai_response is not None:
        st.session_state.transfer_state['ai_response'] = ai_response
    if destination_account is not None:
        st.session_state.transfer_state['destination_account'] = destination_account
    if transfer_amount is not None:
        st.session_state.transfer_state['transfer_amount'] = transfer_amount
    if redirect_url is not None:
        st.session_state.transfer_state['redirect_url'] = redirect_url
    if transaction_result is not None:
        st.session_state.transfer_state['transaction_result'] = transaction_result
    if user_balance is not None:
        st.session_state.transfer_state['user_balance'] = user_balance
    
    # Always ensure user_token is current
    st.session_state.transfer_state['user_token'] = st.session_state.user_token

def get_current_balance(user_token: str = None):
    """Get current balance for the user from database, accounting for any completed transactions"""
    if user_token is None:
        user_token = st.session_state.user_token
    
    # Get actual balance from database
    try:
        db_manager = DatabaseManager("main_DB")
        user_data = db_manager.get_full_user_data(user_token)
        
        if user_data and user_data.get("Account Balance"):
            # Parse the balance string to float
            balance_str = user_data["Account Balance"]
            if balance_str.startswith('$'):
                base_balance = float(balance_str.replace('$', '').replace(',', ''))
            else:
                base_balance = float(balance_str.replace(',', ''))
        else:
            # If no balance found in database, return error
            return "Error: Balance not found"
        
        # Return the actual current balance from Firebase
        # The APIController already updated the balance during transfer processing
        return f"${base_balance:,.2f}"
        
    except Exception as e:
        return f"Error: Could not retrieve balance - {str(e)}"

@st.cache_data
def get_available_users():
    """Get list of available users from the database"""
    try:
        db_manager = DatabaseManager("main_DB")
        user_ids = db_manager.list_all_users()
        
        if not user_ids:
            st.error("No users found in database")
            return {}
        
        # Create display names for users using actual names from database
        available_users = {}
        for user_id in user_ids:
            try:
                # Get user data to extract the actual name
                user_data = db_manager.get_full_user_data(user_id)
                if user_data and user_data.get("User Name"):
                    # Use the actual User Name from database
                    display_name = user_data["User Name"]
                elif user_data and user_data.get("Name"):
                    # Use "Name" field if "User Name" doesn't exist
                    display_name = user_data["Name"]
                else:
                    # Skip users without proper names
                    st.warning(f"User {user_id} has no name field, skipping")
                    continue
                
                available_users[display_name] = user_id
            except Exception as e:
                st.error(f"Error getting user data for {user_id}: {e}")
                continue
        
        return available_users
    except Exception as e:
        st.error(f"Error fetching users from database: {e}")
        return {}

@st.cache_data
def get_user_details(user_token):
    """Get user details from the database"""
    try:
        db_manager = DatabaseManager("main_DB")
        user_data = db_manager.get_full_user_data(user_token)
        
        if user_data:
            return {
                "name": user_data.get("User Name", user_data.get("Name", "Unknown User")),
                "account": user_data.get("Account Number", "No Account Found"),
                "balance": user_data.get("Account Balance", "No Balance Available"),
                "payees": user_data.get("My Payee List", "No Payees Found"),
                "destinations": user_data.get("Destination Accounts", "No Destinations Found"),
                "status": "Active" if user_data else "Inactive"
            }
        else:
            # No fallback - if user not found, return error info
            return {
                "name": "User Not Found",
                "account": "No Account",
                "balance": "No Balance",
                "payees": "No Payees",
                "destinations": "No Destinations",
                "status": "Not Found"
            }
    except Exception as e:
        st.error(f"Database error for user {user_token}: {e}")
        return {
            "name": "Database Error",
            "account": "Database Error",
            "balance": "Database Error", 
            "payees": "Database Error",
            "destinations": "Database Error",
            "status": "Error"
        }


# ─── Streamlit UI Configuration ──────────────────────────────────────

st.set_page_config(
    page_title="LB Finance Unified Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff0000 0%, #cc0000 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    .system-info {
        background-color: #d1ecf1;
        border: 1px solid #b8daff;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ────────────────────────────────────────

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'user_token' not in st.session_state:
    # Get first available user from database instead of hardcoded value
    available_users = get_available_users()
    if available_users:
        st.session_state.user_token = list(available_users.values())[0]
    else:
        st.error("No users found in database. Please check database connection.")
        st.stop()
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"streamlit-session-{st.session_state.user_token}"
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}
if 'transfer_state' not in st.session_state:
    st.session_state.transfer_state = TransferState(
        # User Profile Information
        user_name="",
        user_account="",
        user_balance="",
        user_status="",
        # Transaction Flow Data
        user_query="",
        destination_account="",
        transfer_amount="",
        user_token=st.session_state.user_token,
        ai_response="",
        redirect_url="",
        transaction_result="",
        # Confirmation Management
        needs_confirmation=False,
        confirmation_requested=False,
        user_confirmed=False,
        # State Management Flags
        profile_loaded=False,
        last_balance_check=""
    )

if "voice_audio_processed" not in st.session_state:
    st.session_state["voice_audio_processed"] = False


# ─── Main UI Layout ──────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 LB Finance Unified Agent</h1>
    <h3>Intelligent Banking Assistant with Natural Language Understanding</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with controls and information
with st.sidebar:
    st.header("🔧 Session Controls")
    
    # User Selection
    st.subheader("👤 User Selection")
    
    # Get available users from database
    available_users = get_available_users()
    
    # Check if we have any users
    if not available_users:
        st.error("❌ No users found in database. Please check database connection.")
        st.stop()
    
    # Ensure current user token exists in available users
    current_user_tokens = list(available_users.values())
    if st.session_state.user_token not in current_user_tokens:
        # If current user token is not in available users, set to first available
        st.session_state.user_token = current_user_tokens[0]
    
    # Find current user display name
    current_display_name = None
    for display_name, token in available_users.items():
        if token == st.session_state.user_token:
            current_display_name = display_name
            break
    
    # If not found, use first available user
    if current_display_name is None:
        current_display_name = list(available_users.keys())[0]
    
    selected_user_display = st.selectbox(
        "Select User:",
        options=list(available_users.keys()),
        index=list(available_users.keys()).index(current_display_name),
        help="Choose which user account to use for the session"
    )
    
    # Show number of available users
    st.caption(f"📊 {len(available_users)} user(s) found in database")
    
    # Refresh users button
    if st.button("🔄 Refresh User List", help="Reload user list from database"):
        get_available_users.clear()
        get_user_details.clear()
        st.rerun()
    
    # Get real user details from database
    current_user_token = available_users[selected_user_display]
    user_info = get_user_details(current_user_token)
    
    # Display clean user profile
    display_user_profile(st.session_state.transfer_state)
    
    # Force refresh of user profile display after each conversation
    if st.session_state.chat_history:
        st.empty()  # Clear cache
    
    # Update user token if selection changed
    new_user_token = available_users[selected_user_display]
    if new_user_token != st.session_state.user_token:
        st.session_state.user_token = new_user_token
        # Reset conversation when switching users
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.session_state.conversation_context = {}
        st.session_state.thread_id = f"streamlit-session-{new_user_token}"
        # Reset transfer state with new user token
        st.session_state.transfer_state = TransferState(
            # User Profile Information
            user_name="",
            user_account="",
            user_balance="",
            user_status="",
            # Transaction Flow Data
            user_query="",
            destination_account="",
            transfer_amount="",
            user_token=new_user_token,
            ai_response="",
            redirect_url="",
            transaction_result="",
            # Confirmation Management
            needs_confirmation=False,
            confirmation_requested=False,
            user_confirmed=False,
            # State Management Flags
            profile_loaded=False,
            last_balance_check=""
        )
        # Clear the cache to refresh user details and available users
        get_user_details.clear()
        get_available_users.clear()
        st.success(f"✅ Switched to {selected_user_display}")
        st.rerun()
    
    st.divider()
    
    # Agent Status
    st.success("✅ Using Unified Agent - Natural language understanding & intelligent conversation routing")
    
    st.divider()
    
    # Session Information
    st.info(f"""
    **Session Info:**
    - Session ID: `{st.session_state.thread_id}`
    - User Token: `{st.session_state.user_token}`
    - Current User: {selected_user_display}
    - Messages: {len(st.session_state.chat_history)}
    """)
    
    # Reset conversation button
    if st.button("🔄 Reset Conversation", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.session_state.conversation_context = {}
        st.session_state.thread_id = f"streamlit-session-{st.session_state.user_token}"
        # Reset transfer state
        st.session_state.transfer_state = TransferState(
            # User Profile Information
            user_name="",
            user_account="",
            user_balance="",
            user_status="",
            # Transaction Flow Data
            user_query="",
            destination_account="",
            transfer_amount="",
            user_token=st.session_state.user_token,
            ai_response="",
            redirect_url="",
            transaction_result="",
            # Confirmation Management
            needs_confirmation=False,
            confirmation_requested=False,
            user_confirmed=False,
            # State Management Flags
            profile_loaded=False,
            last_balance_check=""
        )
        st.rerun()
    
    st.divider()
        
    # Testing scenarios
    st.header("🧪 Test Scenarios")
    st.markdown("""
    **Try these examples:**
    
    **🎤 Voice Input (NEW!):**
    - Use **⚡ Quick Voice** for fast recording with defaults
    - Use **🎤 Smart Record** for customizable settings
    - Say: "Send five hundred dollars to Alice"
    - Say: "What is my balance?"
    - Say: "Transfer money to Bob"
    
    **💬 Text Chat:**
    - `hello` → Natural greeting
    - `What services does LB Finance offer?`
    - `Tell me about your branches`
    
    **💰 Transaction Requests:**
    - `Send 500 to Alice` → Complete transfer
    - `to Bob` → Ask for amount
    - `300` → Ask for destination
    - `Transfer money to John`
    
    **🔄 Mixed Conversations:**
    - `What is LB Finance and can I send money?`
    - `I need help with transfer to Sarah`
    """)
    
    # Smart Recording Tips
    st.markdown("""
    **🎯 Smart Recording Tips:**
    - **Speak naturally** - the system detects when you're done
    - **Adjust sensitivity** if it's too sensitive to background noise
    - **Increase silence timeout** if you speak slowly with pauses
    - **Use Quick Voice** for convenience with good default settings
    """)
    
    st.divider()
    
    # Download conversation
    if st.session_state.chat_history:
        conversation_text = "\n".join([
            f"{'User' if msg['type'] == 'user' else 'Bot'}: {msg['content']}"
            for msg in st.session_state.chat_history
        ])
        st.download_button(
            "📥 Download Conversation",
            conversation_text,
            file_name=f"banking_chat_{st.session_state.thread_id}.txt",
            mime="text/plain"
        )

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Banking Chat")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            # Get current user details for welcome message
            current_user = get_user_details(st.session_state.user_token)
            
            st.markdown(f"""
            <div class="system-info">
                👋 Welcome to LB Finance Banking Chatbot, <strong>{current_user['name']}</strong>!<br>
                🏦 Account: {current_user['account']}<br>
                💰 Balance: {current_user['balance']}<br>
                💬 You can ask about services, transfer money, or get account information.
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div class="user-message">
                        👤 You: {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Determine message style based on content
                if "success" in message['content'].lower() and "transferred" in message['content'].lower():
                    message_class = "success-message"
                    icon = "🎉"
                elif "missing" in message['content'].lower() or "provide" in message['content'].lower():
                    message_class = "system-info"
                    icon = "❓"
                elif "not in your" in message['content'].lower() or "insufficient" in message['content'].lower():
                    message_class = "error-message"
                    icon = "❌"
                else:
                    message_class = "bot-message"
                    icon = "🤖"
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div class="{message_class}">
                        {icon} Bot: {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- Voice Command UI ---
    st.markdown(
        """
        <style>
        .voice-btn {
            background: #fff;
            border: 2px solid #007bff;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            margin-right: 8px;
        }
        .voice-btn:hover {
            background: #007bff;
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;align-items:center;">'
        '<span style="font-size:2rem;" class="voice-btn">🎤</span>'
        '<span style="font-size:1rem;">Voice Command</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- NEW: Placeholder for uploader ---
    uploader_ph = st.empty()
    user_input_value = None
    input_type = None

    # Add mic and upload controls as separate buttons
    st.markdown("#### 🎙️ Voice Input Options")
    
    # Smart Recording Information
    st.info("""
    🎯 **New Smart Recording Feature!**
    - **Automatic Detection**: Recording stops automatically when you stop speaking
    - **No More Fixed Duration**: Just speak naturally, and it detects when you're done
    - **Adjustable Settings**: Control sensitivity and silence timeout below
    - **Better User Experience**: No need to worry about timing your speech
    """)
    
    # Check admin status and show warning if needed
    if not is_admin():
        st.warning("⚠️ **Not running as Administrator** - Voice recording may fail with some audio devices")
        st.info("💡 **Tip:** For best voice recording results, restart VS Code or your browser as Administrator")
    
    mic_col, upload_col = st.columns(2)
    with mic_col:
        # Smart recording controls
        st.markdown("**🎤 Smart Recording**")
        silence_limit = st.slider("Silence timeout (seconds)", min_value=2, max_value=10, value=5, key="silence_limit", 
                                help="Recording stops after this many seconds of silence")
        threshold = st.slider("Speech sensitivity", min_value=0.005, max_value=0.1, value=0.02, step=0.005, key="threshold",
                            help="Lower values = more sensitive to quiet speech")
        mic_record_btn = st.button("🎤 Smart Record", key="mic_record_btn", 
                                 help="Records automatically until you stop speaking")
        
        # One-step record and transcribe option
        one_step_btn = st.button("⚡ Quick Voice", key="one_step_btn",
                                help="Record and transcribe in one step with default settings")
        
        # Traditional upload option
        st.markdown("**⬆️ Upload Audio**")
        upload_audio_btn = st.button("📁 Upload Audio File", key="upload_audio_btn")

    #with upload_col:
        #upload_audio_btn = st.button("⬆️ Upload Audio File", key="upload_audio_btn")   #I used one column for both buttons to keep it simple

    # Reset state and show relevant input when a button is pressed
    if mic_record_btn:
        st.session_state["voice_audio_processed"] = False
        st.session_state["audio_mode"] = "mic"
        # Clear any previous error states
        if "transcription_failed" in st.session_state:
            del st.session_state["transcription_failed"]
    if one_step_btn:
        st.session_state["voice_audio_processed"] = False
        st.session_state["audio_mode"] = "one_step"
        # Clear any previous error states
        if "transcription_failed" in st.session_state:
            del st.session_state["transcription_failed"]
    if upload_audio_btn:
        st.session_state["voice_audio_processed"] = False
        st.session_state["audio_mode"] = "upload"
        # Clear any previous error states
        if "transcription_failed" in st.session_state:
            del st.session_state["transcription_failed"]

    # Handle one-step record and transcribe
    if (
        ("audio_mode" in st.session_state and st.session_state["audio_mode"] == "one_step")
        and not st.session_state["voice_audio_processed"]
    ):
        # Show device info
        device_id, device_message = check_audio_devices()
        if device_id is None:
            st.error(f"❌ Audio Error: {device_message}")
            st.error("Please check your microphone connection and permissions.")
            st.session_state["voice_audio_processed"] = True
        else:
            st.info(f"🎤 {device_message}")
            st.info("⚡ Quick Voice: Using default settings (5s silence timeout)")
            
            try:
                with st.spinner("⚡ Recording and transcribing... Speak now!"):
                    # Use the pure in-memory function for better performance
                    voice_transcript = record_and_transcribe_pure_memory()
                
                if voice_transcript:
                    st.success(f"✅ Quick Voice Complete!")
                    st.success(f"📝 Transcription: {voice_transcript}")
                    user_input_value = voice_transcript.strip()
                    input_type = "audio"
                else:
                    st.error("❌ Quick voice failed - no text detected")
                    user_input_value = None
                    
                st.session_state["voice_audio_processed"] = True
                    
            except Exception as e:
                st.error(f"❌ Quick voice error: {e}")
                
                # Check for common permission issues
                error_str = str(e).lower()
                if "permission" in error_str or "access" in error_str or "paerrorcode" in error_str:
                    st.error("🔒 **This appears to be a Windows permissions issue:**")
                    st.error("**SOLUTION:** Try one of these options:")
                    st.error("1. **Close this app and restart VS Code/Terminal as Administrator**")
                    st.error("2. **Or right-click your browser and 'Run as administrator'**")
                    st.error("3. **Check Windows audio permissions for your microphone**")
                    st.error("4. **Use the file upload option below instead**")
                else:
                    st.error("Possible solutions:")
                    st.error("1. Check microphone permissions")
                    st.error("2. Try the regular Smart Record option with custom settings")
                    st.error("3. Use file upload instead")
                    
                st.session_state["voice_audio_processed"] = True
                user_input_value = None

    # Handle mic recording
    if (
        ("audio_mode" in st.session_state and st.session_state["audio_mode"] == "mic")
        and not st.session_state["voice_audio_processed"]
    ):
        # Get user-configured settings
        silence_limit = st.session_state.get("silence_limit", 5.0)
        threshold = st.session_state.get("threshold", 0.02)
        
        # Show device info
        device_id, device_message = check_audio_devices()
        if device_id is None:
            st.error(f"❌ Audio Error: {device_message}")
            st.error("Please check your microphone connection and permissions.")
            st.session_state["voice_audio_processed"] = True
        else:
            st.info(f"🎤 {device_message}")
            st.info(f"📊 Smart Recording Settings: {silence_limit}s silence timeout, {threshold:.3f} sensitivity")
            
            try:
                with st.spinner("🎤 Smart recording in progress... Speak now!"):
                    # Use the new smart recording function with in-memory enhancement
                    audio_path = record_audio_smart(
                        max_wait=30.0,  # Maximum 30 seconds
                        silence_limit=silence_limit,
                        threshold=threshold,
                        apply_enhancement=True  # Apply enhancement during recording
                    )
                
                if audio_path:
                    st.success("✅ Recording completed successfully!")
                    
                    # Transcribe using in-memory enhancement (the file is already enhanced)
                    with st.spinner("🔊 Transcribing enhanced recording..."):
                        voice_transcript = transcribe_uploaded_file(
                            audio_path, apply_enhancement=False  # Already enhanced during recording
                        )
                        
                    if voice_transcript:
                        st.success(f"📝 Transcription: {voice_transcript}")
                        user_input_value = voice_transcript.strip()
                        input_type = "audio"
                        
                        # Clean up the temporary file
                        try:
                            os.remove(audio_path)
                            print(f"🗑️  Cleaned up temporary recording file")
                        except Exception as e:
                            print(f"⚠️  Could not remove temporary file: {e}")
                    else:
                        st.error("❌ Transcription failed - no text detected")
                        user_input_value = None
                        
                else:
                    st.error("❌ Smart recording failed")
                    user_input_value = None
                    
                st.session_state["voice_audio_processed"] = True
                    
            except Exception as e:
                st.error(f"❌ Smart recording error: {e}")
                
                # Check for common permission issues
                error_str = str(e).lower()
                if "permission" in error_str or "access" in error_str or "paerrorcode" in error_str:
                    st.error("🔒 **This appears to be a Windows permissions issue:**")
                    st.error("**SOLUTION:** Try one of these options:")
                    st.error("1. **Close this app and restart VS Code/Terminal as Administrator**")
                    st.error("2. **Or right-click your browser and 'Run as administrator'**")
                    st.error("3. **Check Windows audio permissions for your microphone**")
                    st.error("4. **Use the file upload option below instead**")
                else:
                    st.error("Possible solutions:")
                    st.error("1. Check microphone permissions")
                    st.error("2. Try adjusting sensitivity settings")
                    st.error("3. Use file upload instead")
                    
                st.session_state["voice_audio_processed"] = True
                user_input_value = None

    # Handle file upload
        if user_input_value:
            update_transfer_state(user_query=user_input_value)
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input_value,
                'timestamp': st.session_state.conversation_count
            })
            with st.spinner("🤖 Processing your request..."):
                try:
                    result = execute_banking_graph(
                        thread_id=st.session_state.thread_id,
                        user_input=user_input_value,
                        user_token=st.session_state.user_token,
                        current_state=st.session_state.transfer_state
                    )
                    response = result.get("message", "Sorry, I couldn't process your request.")
                    if result.get("transfer_state"):
                        transfer_state = result["transfer_state"]
                        st.session_state.transfer_state = transfer_state
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and
                            transfer_state.get('transfer_amount')):
                            new_balance = get_current_balance(st.session_state.user_token)
                            update_transfer_state(user_balance=new_balance)
                    st.session_state.conversation_context['last_response'] = result
                    st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and
                        not result.get("needs_more_info", False)):
                        st.session_state.conversation_context = {}
                        update_transfer_state(transaction_result="COMPLETED")
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and 
                        result.get("redirect_url")):
                        response += f" Navigate to: {result['redirect_url']}"
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response,
                        'timestamp': st.session_state.conversation_count
                    })
                    st.session_state.conversation_count += 1
                except Exception as e:
                    error_message = "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment. Please try again in a few moments, or contact our customer support team for immediate assistance."
                    update_transfer_state(ai_response=error_message, transaction_result="ERROR")
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': error_message,
                        'timestamp': st.session_state.conversation_count
                    })
            st.rerun()

    # Handle file upload
    if (
        ("audio_mode" in st.session_state and st.session_state["audio_mode"] == "upload")
        and not st.session_state["voice_audio_processed"]
    ):
        audio_file = uploader_ph.file_uploader(
            "Upload a voice command (WAV/MP3, ≤60s):",
            type=["wav", "mp3"],
            key="voice_audio",
            help="⬆️ Upload and then it will be processed"
        )
        if audio_file is not None:
            with st.spinner("🔊 Transcribing uploaded audio with enhancement..."):
                # Load audio directly into memory for processing
                try:
                    # Read uploaded file into memory
                    audio_bytes = audio_file.read()
                    
                    # Create temporary file for audio loading
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        # Load audio into memory
                        audio_data, sample_rate = load_audio_any(tmp_path, target_sr=48000)
                        
                        # Remove temporary file immediately
                        os.remove(tmp_path)
                        
                        # Transcribe using in-memory processing with enhancement
                        voice_transcript = transcribe_audio_data(
                            audio_data, sample_rate, apply_enhancement=True
                        )
                        
                        if voice_transcript:
                            st.success(f"📝 Transcription: {voice_transcript}")
                            st.info("🔧 Applied voice enhancement to uploaded audio")
                            user_input_value = voice_transcript.strip()
                            input_type = "audio"
                            st.session_state["voice_audio_processed"] = True
                        else:
                            st.error("❌ Transcription failed - no text detected")
                            user_input_value = None
                            
                    except Exception as e:
                        # Clean up temp file on error
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                        raise e
                        
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.error("💡 **Try again:** You can upload a different audio file or try voice recording")
                    user_input_value = None
            
            # Only clear uploader and process if transcription was successful
            if user_input_value:
                uploader_ph.empty()
                # Automatically add to chat and process as user input
                update_transfer_state(user_query=user_input_value)
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input_value,
                    'timestamp': st.session_state.conversation_count
                })
                with st.spinner("🤖 Processing your request..."):
                    try:
                        result = execute_banking_graph(
                            thread_id=st.session_state.thread_id,
                            user_input=user_input_value,
                            user_token=st.session_state.user_token,
                            current_state=st.session_state.transfer_state
                        )
                        response = result.get("message", "Sorry, I couldn't process your request.")
                        if result.get("transfer_state"):
                            transfer_state = result["transfer_state"]
                            st.session_state.transfer_state = transfer_state
                            if (result.get("response_type") == "transaction" and 
                                result.get("status") == "success" and
                                transfer_state.get('transfer_amount')):
                                new_balance = get_current_balance(st.session_state.user_token)
                                update_transfer_state(user_balance=new_balance)
                        st.session_state.conversation_context['last_response'] = result
                        st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and
                            not result.get("needs_more_info", False)):
                            st.session_state.conversation_context = {}
                            update_transfer_state(transaction_result="COMPLETED")
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and 
                            result.get("redirect_url")):
                            response += f" Navigate to: {result['redirect_url']}"
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': response,
                            'timestamp': st.session_state.conversation_count
                        })
                        st.session_state.conversation_count += 1
                    except Exception as e:
                        error_message = "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment. Please try again in a few moments, or contact our customer support team for immediate assistance."
                        update_transfer_state(ai_response=error_message, transaction_result="ERROR")
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': error_message,
                            'timestamp': st.session_state.conversation_count
                        })
                st.rerun()
    # --- END NEW uploader logic ---

    # Unified input form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([4, 1])
        with col_input:
            text_input_value = st.text_input(
                "Type your message:",
                placeholder="e.g., 'hello', 'Send 500 to Alice', or 'What services do you offer?'",
                label_visibility="collapsed",
                value=""
            )
        with col_send:
            send_button = st.form_submit_button("Send 📤", type="primary", use_container_width=True)

    # Unified user input logic
    # user_input_value and input_type already set above for audio

    # Only process if we have a user input (either audio or text)
    if send_button and text_input_value.strip():
        user_input_value = text_input_value.strip()
        input_type = "text"

    if user_input_value and input_type == "text":
        update_transfer_state(user_query=user_input_value)
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input_value,
            'timestamp': st.session_state.conversation_count
        })
        with st.spinner("🤖 Processing your request..."):
            try:
                result = execute_banking_graph(
                    thread_id=st.session_state.thread_id,
                    user_input=user_input_value,
                    user_token=st.session_state.user_token,
                    current_state=st.session_state.transfer_state
                )
                response = result.get("message", "Sorry, I couldn't process your request.")
                if result.get("transfer_state"):
                    transfer_state = result["transfer_state"]
                    st.session_state.transfer_state = transfer_state
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and
                        transfer_state.get('transfer_amount')):
                        new_balance = get_current_balance(st.session_state.user_token)
                        update_transfer_state(user_balance=new_balance)
                st.session_state.conversation_context['last_response'] = result
                st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                if (result.get("response_type") == "transaction" and 
                    result.get("status") == "success" and
                    not result.get("needs_more_info", False)):
                    st.session_state.conversation_context = {}
                    update_transfer_state(transaction_result="COMPLETED")
                if (result.get("response_type") == "transaction" and 
                    result.get("status") == "success" and 
                    result.get("redirect_url")):
                    response += f" Navigate to: {result['redirect_url']}"
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': response,
                    'timestamp': st.session_state.conversation_count
                })
                st.session_state.conversation_count += 1
            except Exception as e:
                error_message = "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment. Please try again in a few moments, or contact our customer support team for immediate assistance."
                update_transfer_state(ai_response=error_message, transaction_result="ERROR")
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': error_message,
                    'timestamp': st.session_state.conversation_count
                })
        st.rerun()

with col2:
    st.header("📊 Statistics")
    
    # Conversation statistics
    total_messages = len(st.session_state.chat_history)
    user_messages = len([msg for msg in st.session_state.chat_history if msg['type'] == 'user'])
    bot_messages = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot'])
    
    # Success/error counts
    success_count = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot' and 'success' in msg['content'].lower()])
    error_count = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot' and ('missing' in msg['content'].lower() or 'insufficient' in msg['content'].lower() or 'not in your' in msg['content'].lower())])
    
    st.metric("Total Messages", total_messages)
    st.metric("User Messages", user_messages)
    st.metric("Bot Messages", bot_messages)
    st.metric("Successful Transfers", success_count, delta=success_count if success_count > 0 else None)
    st.metric("Errors/Requests", error_count, delta=-error_count if error_count > 0 else None)
    
    # Transfer State Tracker
    st.header("🔄 Transfer State")
    
    # Visual state indicators
    state = st.session_state.transfer_state
    
    # Account details
    dest_status = "✅" if state['destination_account'] else "⭕"
    st.markdown(f"{dest_status} **Destination:** {state['destination_account'] or 'Not set'}")
    
    amount_status = "✅" if state['transfer_amount'] else "⭕"
    st.markdown(f"{amount_status} **Amount:** {state['transfer_amount'] or 'Not set'}")
    
    # Confirmation status
    if state.get('needs_confirmation', False) or state.get('confirmation_requested', False):
        if state.get('user_confirmed', False):
            confirm_status = "✅"
            confirm_text = "CONFIRMED"
        elif state.get('confirmation_requested', False):
            confirm_status = "⏳"
            confirm_text = "Awaiting confirmation"
        else:
            confirm_status = "⭕"
            confirm_text = "Needs confirmation"
        st.markdown(f"{confirm_status} **Confirmation:** {confirm_text}")
    
    # Transaction status
    if state['transaction_result']:
        if state['transaction_result'] in ["SUCCESS", "COMPLETED"]:
            st.success(f"🎉 Status: COMPLETED")
        elif state['transaction_result'] == "ERROR":
            st.error(f"❌ Status: {state['transaction_result']}")
        elif state['transaction_result'] == "PENDING_INFO":
            st.warning(f"⏳ Status: {state['transaction_result']}")
        else:
            st.info(f"ℹ️ Status: {state['transaction_result']}")
    else:
        st.markdown("⭕ **Status:** Not started")
    
    # Progress indicator
    # Calculate completed fields - for completed transactions, show all completed
    if state['transaction_result'] in ["SUCCESS", "COMPLETED"]:
        completed_fields = 3  # All fields completed for successful transaction
        total_fields = 3
    else:
        required_fields = [
            bool(state['destination_account']),
            bool(state['transfer_amount'])
        ]
        
        # Add confirmation requirement if needed
        if state.get('needs_confirmation', False) or state.get('confirmation_requested', False):
            required_fields.append(bool(state.get('user_confirmed', False)))
            total_fields = 3
        else:
            total_fields = 2
            
        completed_fields = sum(required_fields)
    
    progress = completed_fields / total_fields if total_fields > 0 else 0
    st.progress(progress, text=f"Transfer Progress: {completed_fields}/{total_fields} steps")
    
    # Quick action buttons
    st.header("⚡ Quick Actions")
    
    quick_actions = [
        ("👋 Say Hello", "hello"),
        ("ℹ️ LB Finance Info", "What services does LB Finance offer?"),
        ("💰 Send $500 to Alice", "Send $500 to Alice"),
        ("👤 To Bob", "to Bob"),
        ("💳 Check Balance", "what is my account balance"),
        ("🏦 Transfer Help", "I need help with transfer"),
        ("❌ Test Invalid Payee", "send money to InvalidAccount"),
        ("💸 Transfer 200000 to Mike", "transfer 200000 to Mike")
    ]
    
    for label, action in quick_actions:
        if st.button(label, key=f"quick_{action}", use_container_width=True):
            # Simulate user input
            st.session_state.chat_history.append({
                'type': 'user',
                'content': action,
                'timestamp': st.session_state.conversation_count
            })
            
            # Process the action
            with st.spinner("Processing..."):
                try:
                    # Use unified agent
                    result = execute_banking_graph(
                        thread_id=st.session_state.thread_id,
                        user_input=action,
                        user_token=st.session_state.user_token,
                        current_state=st.session_state.transfer_state  # Pass current conversation state
                    )
                    response = result.get("message", "Sorry, I couldn't process your request.")
                    
                    # Update transfer state with the full state from graph
                    if result.get("transfer_state"):
                        transfer_state = result["transfer_state"]
                        # Update the complete state directly - preserve the transaction_result from the graph
                        st.session_state.transfer_state = transfer_state
                        
                        # Update balance if transfer was successful
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and
                            transfer_state.get('transfer_amount')):
                            new_balance = get_current_balance(st.session_state.user_token)
                            update_transfer_state(user_balance=new_balance)
                    
                    # Update conversation context with graph result
                    st.session_state.conversation_context['last_response'] = result
                    st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                    
                    # Reset context after successful transactions
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and
                        not result.get("needs_more_info", False)):
                        st.session_state.conversation_context = {}
                    
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and 
                        result.get("redirect_url")):
                        response += f" Navigate to: {result['redirect_url']}"
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response,
                        'timestamp': st.session_state.conversation_count
                    })
                    
                    st.session_state.conversation_count += 1
                    
                except Exception as e:
                    # Generate formal error response for quick actions too
                    error_message = "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment. Please try again in a few moments, or contact our customer support team for immediate assistance."
                    update_transfer_state(ai_response=error_message, transaction_result="ERROR")
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': error_message,
                        'timestamp': st.session_state.conversation_count
                    })
            
            st.rerun()
# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;">
    <strong>LB Finance Unified Agent</strong> — Intelligent Banking Assistant<br>
    Built with Streamlit | Powered by Unified Agent Platform<br>
    Leveraging Natural Language Understanding and Secure Transaction Processing<br><br>
    <em>This application is currently in experimental development and may be subject to changes.</em><br>
    &copy; 2025 LB Finance PLC. All rights reserved.
</div>
""", unsafe_allow_html=True)