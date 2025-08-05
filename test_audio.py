#!/usr/bin/env python3
"""
Audio Device Test Script
Test audio devices and recording functionality
"""

import os
import sys
import time
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_audio_system():
    """Comprehensive audio system test."""
    print("🎤 Audio System Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Import libraries
    print("\n1️⃣ Testing library imports...")
    try:
        import sounddevice as sd
        import soundfile as sf
        import librosa
        print("✅ All audio libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Check audio devices
    print("\n2️⃣ Checking audio devices...")
    try:
        from functions import check_audio_devices, find_working_audio_device
        
        device_works, devices, default_input = check_audio_devices()
        
        if device_works:
            print("✅ Default audio device works")
        else:
            print("❌ Default audio device failed, searching for alternatives...")
            working_device_id, working_device = find_working_audio_device()
            if working_device:
                print(f"✅ Found working device: {working_device['name']}")
                # Set as default for testing
                sd.default.device = working_device_id
            else:
                print("❌ No working audio devices found")
                return False
                
    except Exception as e:
        print(f"❌ Device check failed: {e}")
        return False
    
    # Test 3: Simple recording test
    print("\n3️⃣ Testing simple recording...")
    try:
        print("🔴 Recording 3 seconds of audio (speak now)...")
        duration = 3  # seconds
        sample_rate = 16000
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()  # Wait for recording to complete
        
        # Check if we got audio data
        if audio_data is not None and len(audio_data) > 0:
            max_amplitude = np.max(np.abs(audio_data))
            print(f"✅ Recording successful! Max amplitude: {max_amplitude:.4f}")
            
            # Save test recording
            test_filename = f"test_recording_{int(time.time())}.wav"
            sf.write(test_filename, audio_data, sample_rate)
            print(f"💾 Test recording saved as: {test_filename}")
            
            return True
        else:
            print("❌ Recording failed - no audio data captured")
            return False
            
    except Exception as e:
        print(f"❌ Recording test failed: {e}")
        return False

def test_smart_recording():
    """Test the smart recording function."""
    print("\n4️⃣ Testing smart recording function...")
    try:
        from functions import record_audio_smart
        
        print("🎙️ Testing smart recording (speak for a few seconds)...")
        audio_file = record_audio_smart(
            sample_rate=16000,
            max_wait=10.0,
            silence_limit=3.0,
            threshold=0.02
        )
        
        if audio_file and os.path.exists(audio_file):
            print(f"✅ Smart recording successful: {audio_file}")
            return True
        else:
            print("❌ Smart recording failed")
            return False
            
    except Exception as e:
        print(f"❌ Smart recording test failed: {e}")
        return False

def test_transcription():
    """Test transcription functionality."""
    print("\n5️⃣ Testing transcription...")
    try:
        from functions import transcribe_uploaded_file
        
        # Look for existing audio files to test transcription
        audio_files = [f for f in os.listdir('.') if f.endswith('.wav')]
        
        if audio_files:
            test_file = audio_files[0]
            print(f"🔄 Testing transcription with: {test_file}")
            
            transcription = transcribe_uploaded_file(test_file)
            
            if transcription and transcription.strip():
                print(f"✅ Transcription successful: '{transcription}'")
                return True
            else:
                print("❌ Transcription failed - no text generated")
                return False
        else:
            print("⚠️ No audio files found for transcription test")
            return True  # Not a failure, just no files to test
            
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

def main():
    """Run all audio tests."""
    print("🎤 Starting comprehensive audio system test...")
    
    tests = [
        ("Audio System Check", test_audio_system),
        ("Smart Recording", test_smart_recording),
        ("Transcription", test_transcription)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Audio system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("\n💡 Common solutions:")
        print("   - Run as Administrator")
        print("   - Check microphone permissions")
        print("   - Update audio drivers")
        print("   - Try different audio device")
        print("   - Restart Windows Audio service")

if __name__ == "__main__":
    main()
