"""
Streamlit Audio Interface
Direct microphone recording interface for Streamlit apps
Replaces browser recording with working microphone recording
"""
import streamlit as st
import time
from functions import record_audio_smart, transcribe_uploaded_file
import os

def streamlit_microphone_interface():
    """
    Streamlit interface for direct microphone recording
    Returns transcribed text if successful, None otherwise
    """
    
    # Initialize session state
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = "ready"
    if 'recording_error' not in st.session_state:
        st.session_state.recording_error = None
    if 'recorded_file' not in st.session_state:
        st.session_state.recorded_file = None
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None
    if 'transcription_returned' not in st.session_state:
        st.session_state.transcription_returned = False
    if 'manual_stop_requested' not in st.session_state:
        st.session_state.manual_stop_requested = False
    
    # State machine for recording flow
    status = st.session_state.recording_status
    
    # If we have a completed transcription that hasn't been returned yet, return it
    if (status == "completed" and 
        st.session_state.transcription_result and 
        not st.session_state.transcription_returned):
        
        # Mark as returned and reset for next recording
        st.session_state.transcription_returned = True
        
        # Show result briefly
        st.success("✅ Voice message processed!")
        st.success(f"📝 **Your message:** {st.session_state.transcription_result}")
        
        # Reset states for next recording after a delay
        transcription_to_return = st.session_state.transcription_result
        
        # Reset immediately so user can start new recording
        st.session_state.recording_status = "ready"
        st.session_state.recorded_file = None
        st.session_state.transcription_result = None
        st.session_state.transcription_returned = False
        st.session_state.recording_error = None
        st.session_state.manual_stop_requested = False
        
        return transcription_to_return
    
    if status == "ready":
        # Show start recording button
        st.info("🎤 Ready to record. Click the button to start.")
        
        if st.button("🎤 Start Microphone Recording", type="primary", key="start_mic_recording"):
            # Reset previous states
            st.session_state.recording_error = None
            st.session_state.recorded_file = None
            st.session_state.transcription_result = None
            st.session_state.transcription_returned = False
            st.session_state.manual_stop_requested = False
            st.session_state.recording_status = "recording"
            st.rerun()
            
    elif status == "recording":
        # Show recording in progress with stop button
        st.warning("🔴 Recording in progress...")
        st.info("🎙️ Speak now! Recording will auto-stop after 5 seconds of silence or 30 seconds total.")
        
        # Add manual stop button
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("") # spacing
        with col2:
            if st.button("🛑 STOP Recording", type="secondary", key="stop_recording"):
                st.session_state.manual_stop_requested = True
                st.info("🛑 Stop requested... finishing recording...")
                time.sleep(0.5)  # Brief delay to ensure stop signal is processed
                st.rerun()
        
        # Perform the actual recording
        try:
            with st.spinner("🎤 Recording audio..."):
                # Function to check if manual stop was requested
                def check_manual_stop():
                    return st.session_state.get('manual_stop_requested', False)
                
                audio_file = record_audio_smart(
                    sample_rate=16000,
                    max_wait=30.0,      # 30 seconds max
                    silence_limit=5.0,  # 5 seconds of silence
                    threshold=0.02,
                    apply_enhancement=False,  # We'll enhance during transcription
                    stop_check_func=check_manual_stop
                )
            
            if audio_file and os.path.exists(audio_file):
                st.session_state.recorded_file = audio_file
                st.session_state.recording_status = "transcribing"
                st.success("✅ Recording completed!")
                st.rerun()
            else:
                st.session_state.recording_error = "❌ Recording failed - no audio captured"
                st.session_state.recording_status = "error"
                st.rerun()
                
        except Exception as e:
            st.session_state.recording_error = f"❌ Recording error: {e}"
            st.session_state.recording_status = "error"
            st.rerun()
            
    elif status == "transcribing":
        # Show transcription in progress
        st.success("✅ Recording completed!")
        
        if st.session_state.recorded_file:
            try:
                with st.spinner("🔄 Transcribing audio..."):
                    transcription = transcribe_uploaded_file(
                        st.session_state.recorded_file,
                        apply_enhancement=True
                    )
                
                if transcription and transcription.strip():
                    st.session_state.transcription_result = transcription
                    st.session_state.recording_status = "completed"
                    
                    # Clean up the audio file
                    try:
                        os.remove(st.session_state.recorded_file)
                    except:
                        pass
                    
                    st.rerun()
                else:
                    st.session_state.recording_error = "❌ Transcription failed - no text generated"
                    st.session_state.recording_status = "error"
                    st.rerun()
                    
            except Exception as e:
                st.session_state.recording_error = f"❌ Transcription error: {e}"
                st.session_state.recording_status = "error"
                st.rerun()
                
    elif status == "completed":
        # This state should be very brief - just to trigger the return above
        # If we get here without returning, something went wrong, reset
        st.session_state.recording_status = "ready"
        st.rerun()
            
    elif status == "error":
        # Show error and retry option
        if st.session_state.recording_error:
            st.error(st.session_state.recording_error)
        
        if st.button("🔄 Try Again", key="retry_recording"):
            st.session_state.recording_status = "ready"
            st.session_state.recording_error = None
            st.session_state.manual_stop_requested = False
            st.rerun()
    
    return None


def microphone_record_and_transcribe(**kwargs):
    """
    Drop-in replacement for browser recording functions.
    Uses direct microphone recording instead of browser recording.
    """
    return streamlit_microphone_interface()
