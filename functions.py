## functions.py
#import os
#import torch
#import librosa
#from transformers import WhisperForConditionalGeneration, WhisperProcessor
#
#
#
## Cache the model & processor at import time
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
#processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
#
#def transcribe(audio_path: str) -> str:
#    """
#    Load an audio file and return Whisper’s transcription.
#    """
#    # 1. Load & resample
#    speech, sr = librosa.load(audio_path, sr=16000)
#    # 2. Feature-extract
#    inputs = processor.feature_extractor(
#        speech, sampling_rate=16000, return_tensors="pt"
#    )
#    attention_mask = torch.ones_like(inputs["input_features"]).long()
#    # 3. Inference
#    with torch.no_grad():
#        predicted_ids = model.generate(
#            inputs["input_features"],
#            attention_mask=attention_mask,
#            language="en",
#        )
#    # 4. Decode and return
#    return processor.tokenizer.batch_decode(
#        predicted_ids, skip_special_tokens=True
#    )[0]
#
#


#import torch
#from faster_whisper import WhisperModel
#
## Choose model size; "medium" is a good quality/speed tradeoff for your use case.
#MODEL_SIZE = "tiny"
#device = "cuda" if torch.cuda.is_available() else "cpu"
#
## You can use compute_type="int8" to reduce memory if needed (with small quality tradeoff)
#model = WhisperModel(MODEL_SIZE, device=device, compute_type="float32")
#
#def transcribe(audio_path: str, language: str = "en") -> str:
#    """
#    Transcribe audio and return the full concatenated text.
#    """
#    segments, info = model.transcribe(
#        audio_path,
#        beam_size=5,
#        language=language,
#        vad_filter=True  # optional: filters out silence / non-speech
#    )
#    # You can also return per-segment timestamps if needed
#    full_text = " ".join(segment.text for segment in segments)
#    return full_text
#



import os
import time
import torch
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime
import sounddevice as sd

from faster_whisper import WhisperModel

# === CONFIG (edit these manually) ===
#INPUT_PATH       = r"send_500_rs_to_hnb.mp3"
MODEL_SIZE       = "tiny"       # choices: tiny, base, small, medium, large
COMPUTE_TYPE     = "float32"    # choices: float16, int8, float32
LANGUAGE         = "en"         # or None for autodetect
ENABLE_ENHANCEMENT = True       # set False to skip enhancement entirely
VAD_FILTER       = True         # voice activity detection filtering

# === SMART RECORDING CONFIG ===
MAX_WAIT = 30.0        # Maximum total wait time (seconds)
SILENCE_LIMIT = 5.0    # Silence duration to trigger stop (seconds)
BLOCK_DURATION = 0.1   # Block size in seconds
RECORDING_SAMPLE_RATE = 16000    # Samples per second for recording
THRESHOLD = 0.02       # RMS threshold for detecting speech
# ====================================

# Cache for working audio device to avoid repeated testing
_cached_working_device = None

ENHANCED_OUTPUT_DIR = r".\voice_enhancement"
os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)

# --- ClearVoice setup (optional) ---
try:
    from clearvoice import ClearVoice
    HAVE_CLEARVOICE = True
except ImportError:
    HAVE_CLEARVOICE = False
    print("Warning: ClearVoice not installed; enhancement will be skipped.")

def init_clearvoice():
    if not HAVE_CLEARVOICE:
        return None
    return ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

_myClearVoice = init_clearvoice()


def check_audio_devices():
    """Check available audio devices and return device info."""
    try:
        import sounddevice as sd
        print("🔍 Checking available audio devices...")
        
        # Get device list
        devices = sd.query_devices()
        print(f"📱 Found {len(devices)} audio devices:")
        
        default_input = sd.default.device[0] if hasattr(sd.default, 'device') else None
        default_output = sd.default.device[1] if hasattr(sd.default, 'device') else None
        
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("INPUT")
            if device['max_output_channels'] > 0:
                device_type.append("OUTPUT")
            
            default_marker = ""
            if i == default_input:
                default_marker += " [DEFAULT INPUT]"
            if i == default_output:
                default_marker += " [DEFAULT OUTPUT]"
            
            print(f"  {i}: {device['name']} - {'/'.join(device_type)}{default_marker}")
            print(f"      Max channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            print(f"      Sample rate: {device['default_samplerate']} Hz")
        
        # Test default input device
        if default_input is not None:
            print(f"🎤 Testing default input device: {devices[default_input]['name']}")
            try:
                # Try a very short test recording
                test_duration = 0.1  # 100ms test
                test_data = sd.rec(int(test_duration * 16000), samplerate=16000, channels=1, dtype=np.float32)
                sd.wait()
                print("✅ Default input device test successful")
                return True, devices, default_input
            except Exception as e:
                print(f"❌ Default input device test failed: {e}")
                return False, devices, default_input
        else:
            print("❌ No default input device found")
            return False, devices, None
            
    except Exception as e:
        print(f"❌ Audio device check failed: {e}")
        return False, [], None


def find_working_audio_device():
    """Find a working audio input device."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Try each input device
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"🧪 Testing device {i}: {device['name']}")
                try:
                    # Set this device as default temporarily
                    sd.default.device = i
                    
                    # Very short test
                    test_data = sd.rec(int(0.1 * 16000), samplerate=16000, channels=1, dtype=np.float32)
                    sd.wait()
                    
                    print(f"✅ Device {i} works: {device['name']}")
                    return i, device
                    
                except Exception as e:
                    print(f"❌ Device {i} failed: {e}")
                    continue
        
        print("❌ No working audio input devices found")
        return None, None
        
    except Exception as e:
        print(f"❌ Error finding working device: {e}")
        return None, None


def reset_audio_device_cache():
    """Reset the cached working audio device."""
    global _cached_working_device
    _cached_working_device = None
    print("🔄 Audio device cache reset")


def record_audio_smart(sample_rate=RECORDING_SAMPLE_RATE, max_wait=MAX_WAIT, 
                      silence_limit=SILENCE_LIMIT, threshold=THRESHOLD,
                      output_dir=None, apply_enhancement=True, stop_check_func=None):
    """
    Smart audio recording with automatic silence detection.
    Records until silence is detected or max time is reached.
    
    Args:
        sample_rate: Audio sample rate for recording
        max_wait: Maximum recording time in seconds
        silence_limit: Silence duration before stopping in seconds
        threshold: RMS threshold for detecting speech
        output_dir: Directory to save the recorded file (optional)
        apply_enhancement: Whether to apply enhancement after recording
        stop_check_func: Function to check if manual stop was requested (returns bool)
    
    Returns:
        str: Path to the recorded audio file, or None if failed
    """
    global _cached_working_device
    
    print("🎙️ Starting smart microphone recording...")
    print(f"📊 Config: {sample_rate}Hz, max {max_wait}s, silence limit {silence_limit}s")
    
    # Import sounddevice at the start to ensure it's available throughout the function
    import sounddevice as sd
    
    # Use cached device if available, otherwise find working device
    if _cached_working_device is not None:
        print(f"🔄 Using cached working device: {_cached_working_device}")
        try:
            # Quick test of cached device
            sd.default.device = _cached_working_device
            test_data = sd.rec(int(0.1 * 16000), samplerate=16000, channels=1, dtype=np.float32)
            sd.wait()
            print("✅ Cached device still works")
        except Exception as e:
            print(f"⚠️ Cached device failed ({e}), searching for new device...")
            _cached_working_device = None
    
    if _cached_working_device is None:
        # First, check audio devices
        device_works, devices, default_input = check_audio_devices()
        
        if not device_works:
            print("🔧 Default device failed, searching for working device...")
            working_device_id, working_device = find_working_audio_device()
            
            if working_device_id is not None:
                print(f"✅ Found working device: {working_device['name']}")
                # Set the working device as default and cache it
                sd.default.device = working_device_id
                _cached_working_device = working_device_id
            else:
                print("❌ No working audio devices found")
                print("💡 Troubleshooting tips:")
                print("   1. Check if your microphone is connected and enabled")
                print("   2. Ensure microphone permissions are granted to Python/Streamlit")
                print("   3. Try running as Administrator")
                print("   4. Check Windows Sound settings")
                print("   5. Update audio drivers")
                return None
        else:
            # Cache the working default device
            _cached_working_device = default_input
    
    try:
        # Initialize recording variables
        recorded_data = []
        silent_chunks = 0
        total_chunks = 0
        max_chunks = int(max_wait / BLOCK_DURATION)
        silence_chunks_needed = int(silence_limit / BLOCK_DURATION)
        chunk_size = int(sample_rate * BLOCK_DURATION)
        
        print(f"🔴 Recording started... Speak now! (threshold: {threshold})")
        print("💡 Recording will stop automatically after silence or max time")
        
        # Start recording
        recording_started = False
        recording_complete = False
        
        def audio_callback(indata, frames, time, status):
            nonlocal recorded_data, silent_chunks, total_chunks, recording_started, recording_complete
            if status:
                print(f"⚠️  Audio callback status: {status}")
            
            # Don't process if recording is complete
            if recording_complete:
                return
            
            # Convert to mono if stereo
            audio_chunk = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            recorded_data.append(audio_chunk.copy())
            
            # Calculate RMS for this chunk
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            total_chunks += 1
            
            if rms > threshold:
                silent_chunks = 0
                if not recording_started:
                    recording_started = True
                    print("🎤 Speech detected, recording active...")
                print(f"🟢 Audio level: {rms:.4f}")
            else:
                silent_chunks += 1
                if recording_started:
                    print(f"🔵 Silence: {silent_chunks}/{silence_chunks_needed} chunks")
                    
                # Check if we should stop due to silence
                if silent_chunks >= silence_chunks_needed:
                    print(f"🛑 Silence limit reached, marking recording complete")
                    recording_complete = True
            
            # Check if we should stop due to max time
            if total_chunks >= max_chunks:
                print(f"🛑 Max time reached, marking recording complete")
                recording_complete = True
        
        # Try different audio configurations if the first fails
        audio_configs = [
            # Standard config
            {'samplerate': sample_rate, 'channels': 1, 'dtype': np.float32, 'blocksize': chunk_size},
            # Try different sample rates
            {'samplerate': 44100, 'channels': 1, 'dtype': np.float32, 'blocksize': int(44100 * BLOCK_DURATION)},
            {'samplerate': 48000, 'channels': 1, 'dtype': np.float32, 'blocksize': int(48000 * BLOCK_DURATION)},
            # Try different data types
            {'samplerate': sample_rate, 'channels': 1, 'dtype': 'int16', 'blocksize': chunk_size},
            # Try without explicit blocksize
            {'samplerate': sample_rate, 'channels': 1, 'dtype': np.float32},
        ]
        
        recording_successful = False
        
        for i, config in enumerate(audio_configs):
            try:
                print(f"🧪 Trying audio configuration {i+1}/{len(audio_configs)}: {config}")
                
                # Reset for this configuration attempt (but keep variables that callback uses)
                if i > 0:  # Don't reset on first attempt
                    recorded_data = []
                    silent_chunks = 0
                    total_chunks = 0
                    recording_started = False
                    recording_complete = False
                
                # Record audio with callback
                with sd.InputStream(callback=audio_callback, **config):
                    # Monitor recording progress
                    actual_sample_rate = config['samplerate']
                    actual_block_duration = BLOCK_DURATION
                    
                    while not recording_complete and total_chunks < max_chunks:
                        time.sleep(actual_block_duration)
                        
                        # Check for manual stop signal
                        if stop_check_func and stop_check_func():
                            print(f"🛑 Stopping: Manual stop requested")
                            recording_complete = True
                            break
                        
                        # Show progress every second
                        if total_chunks % int(1.0 / actual_block_duration) == 0:
                            elapsed = total_chunks * actual_block_duration
                            print(f"⏱️  Recording: {elapsed:.1f}s / {max_wait}s")
                    
                    if total_chunks >= max_chunks:
                        print(f"🛑 Stopping: Maximum recording time ({max_wait}s) reached")
                
                recording_successful = True
                used_sample_rate = actual_sample_rate
                break
                
            except Exception as e:
                print(f"❌ Configuration {i+1} failed: {e}")
                # Reset variables for next attempt
                recorded_data = []
                silent_chunks = 0
                total_chunks = 0
                recording_started = False
                recording_complete = False
                continue
        
        if not recording_successful:
            print("❌ All audio configurations failed")
            return None
        
        # Process recorded data
        print(f"🔍 Processing recorded data: {len(recorded_data)} chunks")
        if not recorded_data:
            print("❌ No audio data recorded")
            return None
        
        # Debug information about recorded chunks
        total_samples = sum(len(chunk) for chunk in recorded_data)
        print(f"📊 Total audio samples collected: {total_samples}")
        print(f"📊 Total chunks collected: {len(recorded_data)}")
        print(f"📊 Expected samples per chunk: {chunk_size}")
        
        # Concatenate all chunks
        try:
            full_audio = np.concatenate(recorded_data)
            print(f"✅ Successfully concatenated {len(recorded_data)} chunks into {len(full_audio)} samples")
        except Exception as e:
            print(f"❌ Failed to concatenate audio chunks: {e}")
            return None
        
        # Convert data type if needed
        if full_audio.dtype == 'int16':
            full_audio = full_audio.astype(np.float32) / 32768.0
        
        # Resample if needed
        if used_sample_rate != sample_rate:
            print(f"🔄 Resampling from {used_sample_rate}Hz to {sample_rate}Hz")
            full_audio = librosa.resample(full_audio, orig_sr=used_sample_rate, target_sr=sample_rate)
        
        duration = len(full_audio) / sample_rate
        print(f"✅ Recording complete: {duration:.2f}s, {len(full_audio)} samples")
        
        # Generate output filename
        if output_dir is None:
            output_dir = ENHANCED_OUTPUT_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"recording_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the recording
        sf.write(output_path, full_audio, sample_rate, subtype="PCM_16")
        print(f"💾 Audio saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None


def load_audio_any(path: str, target_sr: int = 48000):
    """
    Robust audio loader: try librosa first, then fall back to pydub.
    Returns (y, sr) as float32 mono.
    """
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype("float32"), sr
    except Exception as e:
        print(f"[audio load] librosa failed ({e}), trying pydub fallback.")
        from pydub import AudioSegment  # ensure pydub is installed if you need this
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        max_int = float(1 << (8 * audio.sample_width - 1))
        y = samples / max_int
        return y.astype("float32"), target_sr


def _spectral_gate(y: np.ndarray, sr: int, n_fft=1024, hop_length=256, prop_decrease=1.0):
    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S_full), np.angle(S_full)
    frame_energies = np.sum(magnitude, axis=0)
    thresh = np.percentile(frame_energies, 10)
    noise_frames = magnitude[:, frame_energies <= thresh]
    noise_profile = (np.mean(noise_frames, axis=1, keepdims=True)
                     if noise_frames.size else
                     np.mean(magnitude, axis=1, keepdims=True))
    mask_gain = np.clip((magnitude - noise_profile) / (magnitude + 1e-8), 0, 1) ** prop_decrease
    enhanced_mag = magnitude * mask_gain
    y_enhanced = librosa.istft(enhanced_mag * np.exp(1j * phase), hop_length=hop_length)
    y_enhanced = y_enhanced / (np.max(np.abs(y_enhanced)) + 1e-9) * 0.99
    return y_enhanced


def enhance_audio_fallback(input_path: str) -> str:
    y, sr = load_audio_any(input_path, target_sr=48000)
    try:
        import noisereduce as nr
        noise_sample = y[: min(len(y), int(0.5 * sr))]
        enhanced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
        print("[enhancement-fallback] Used noisereduce")
    except ImportError:
        enhanced = _spectral_gate(y, sr, prop_decrease=1.0)
        print("[enhancement-fallback] Used spectral gating fallback")

    base = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"{base}_enhanced_fallback_{ts}.wav"
    out_path = os.path.join(ENHANCED_OUTPUT_DIR, out_fname)
    sf.write(out_path, enhanced.astype("float32"), sr, subtype="PCM_16")
    print(f"[enhancement-fallback] Saved to: {out_path}")
    return out_path


def voice_enhancement_in_memory(audio_data: np.ndarray, sample_rate: int = 48000) -> tuple:
    """
    Apply voice enhancement to audio data in memory without saving files.
    
    Args:
        audio_data: Raw audio data as numpy array
        sample_rate: Sample rate of the audio
    
    Returns:
        tuple: (enhanced_audio_data, sample_rate) or (original_audio_data, sample_rate) if enhancement fails
    """
    if not ENABLE_ENHANCEMENT:
        print("[enhancement] Enhancement disabled, returning original audio")
        return audio_data, sample_rate
    
    # Ensure audio is in correct format
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0] if audio_data.shape[1] == 1 else np.mean(audio_data, axis=1)
    
    # Normalize input audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
    
    # Try ClearVoice enhancement first
    if _myClearVoice:
        try:
            # Save to temporary file for ClearVoice (it requires file input)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data.astype("float32"), sample_rate, subtype="PCM_16")
                tmp_path = tmp_file.name
            
            try:
                enhanced_audio = _myClearVoice(input_path=tmp_path, online_write=False)
                
                if enhanced_audio is None or not hasattr(enhanced_audio, "ndim"):
                    raise ValueError("ClearVoice returned no usable audio")

                # Handle channels & normalization
                if enhanced_audio.ndim == 2:
                    enhanced_audio = (enhanced_audio.squeeze(0)
                                      if enhanced_audio.shape[0] == 1
                                      else enhanced_audio.squeeze(1))
                
                # Normalize output
                peak = np.max(np.abs(enhanced_audio))
                if peak > 1.0:
                    enhanced_audio = enhanced_audio / peak * 0.99

                print("[enhancement] Successfully applied ClearVoice enhancement in memory")
                return enhanced_audio.astype("float32"), 48000
                
            finally:
                # Clean up temporary file
                try:
                    os.remove(tmp_path)
                except:
                    pass

        except Exception as e:
            print(f"[enhancement] ClearVoice failed ({e}), falling back to local enhancement")

    # Fallback to local enhancement methods
    try:
        # Try noisereduce first
        try:
            import noisereduce as nr
            # Use first 0.5 seconds as noise sample
            noise_sample_length = min(len(audio_data), int(0.5 * sample_rate))
            noise_sample = audio_data[:noise_sample_length]
            enhanced = nr.reduce_noise(y=audio_data, sr=sample_rate, y_noise=noise_sample, prop_decrease=1.0)
            print("[enhancement] Applied noisereduce enhancement in memory")
            return enhanced.astype("float32"), sample_rate
            
        except ImportError:
            # Fallback to spectral gating
            enhanced = _spectral_gate(audio_data, sample_rate, prop_decrease=1.0)
            print("[enhancement] Applied spectral gating enhancement in memory")
            return enhanced.astype("float32"), sample_rate
            
    except Exception as e:
        print(f"[enhancement] All enhancement methods failed ({e}), returning original audio")
        return audio_data, sample_rate


def voice_enhancement(filename: str) -> str:
    """
    Legacy function for file-based enhancement (kept for backward compatibility).
    Try ClearVoice; if it fails or returns invalid output, fall back to local denoising.
    """
    if not ENABLE_ENHANCEMENT:
        return filename

    # Load audio file
    try:
        audio_data, original_sr = load_audio_any(filename, target_sr=48000)
        
        # Apply in-memory enhancement
        enhanced_audio, enhanced_sr = voice_enhancement_in_memory(audio_data, original_sr)
        
        # Save enhanced audio to file
        base = os.path.splitext(os.path.basename(filename))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fname = f"{base}_enhanced_memory_{ts}.wav"
        out_path = os.path.join(ENHANCED_OUTPUT_DIR, out_fname)
        sf.write(out_path, enhanced_audio, enhanced_sr, subtype="PCM_16")
        print(f"[enhancement] Saved enhanced audio to: {out_path}")
        return out_path
        
    except Exception as e:
        print(f"[enhancement] Failed to enhance {filename}: {e}")
        return filename


# ─── Whisper model is initialized ONCE here ────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_model = WhisperModel(MODEL_SIZE, device=device, compute_type=COMPUTE_TYPE)


def transcribe_audio_data(audio_data: np.ndarray, sample_rate: int, 
                          language: str = LANGUAGE, vad_filter: bool = VAD_FILTER,
                          apply_enhancement: bool = True) -> str:
    """
    Transcribe audio data directly from memory without saving to file.
    
    Args:
        audio_data: Raw audio data as numpy array
        sample_rate: Sample rate of the audio
        language: Language for transcription
        vad_filter: Voice activity detection filtering
        apply_enhancement: Whether to apply voice enhancement
    
    Returns:
        str: Transcribed text, or None if failed
    """
    print(f"[transcribe] Processing audio data: {len(audio_data)} samples at {sample_rate}Hz")
    start_total = time.time()
    
    try:
        # Apply voice enhancement in memory if requested
        if apply_enhancement:
            print("🔧 Applying voice enhancement to audio data...")
            enhanced_audio, enhanced_sr = voice_enhancement_in_memory(audio_data, sample_rate)
            final_audio = enhanced_audio
            final_sr = enhanced_sr
        else:
            final_audio = audio_data
            final_sr = sample_rate
        
        # Create temporary file for Whisper (Whisper requires file input)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, final_audio, final_sr, subtype="PCM_16")
            tmp_path = tmp_file.name
        
        try:
            print(f"[transcribe] Running Whisper transcription...")
            t0 = time.time()
            
            segments, info = _whisper_model.transcribe(
                tmp_path,
                beam_size=5,
                language=language,
                vad_filter=vad_filter,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            
            t1 = time.time()
            
            print("\n=== Final Transcription ===")
            print(text, "\n")
            print(f"Transcription time: {t1 - t0:.2f}s")
            print(f"Total elapsed: {time.time() - start_total:.2f}s")
            
            return text
            
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"[transcribe] Failed: {e}")
        return None


def transcribe_uploaded_file(file_path: str, language: str = LANGUAGE, 
                           vad_filter: bool = VAD_FILTER, apply_enhancement: bool = True) -> str:
    """
    Transcribe an uploaded audio file with in-memory enhancement.
    
    Args:
        file_path: Path to the uploaded audio file
        language: Language for transcription
        vad_filter: Voice activity detection filtering
        apply_enhancement: Whether to apply voice enhancement
    
    Returns:
        str: Transcribed text, or None if failed
    """
    if not os.path.isfile(file_path):
        print(f"Input file does not exist: {file_path}")
        return None
    
    print(f"Input audio file: {file_path}")
    
    try:
        # Load audio file into memory
        audio_data, sample_rate = load_audio_any(file_path, target_sr=48000)
        
        # Use the in-memory transcription function
        return transcribe_audio_data(audio_data, sample_rate, language, vad_filter, apply_enhancement)
        
    except Exception as e:
        print(f"[transcribe] Failed to process uploaded file: {e}")
        return None


def transcribe(audio_path: str, language: str = LANGUAGE, vad_filter: bool = VAD_FILTER) -> str:
    """
    Legacy transcribe function (kept for backward compatibility).
    Now uses in-memory enhancement for better performance.
    """
    return transcribe_uploaded_file(audio_path, language, vad_filter, apply_enhancement=True)


def record_and_transcribe(sample_rate=RECORDING_SAMPLE_RATE, max_wait=MAX_WAIT,
                         silence_limit=SILENCE_LIMIT, threshold=THRESHOLD,
                         language=LANGUAGE, vad_filter=VAD_FILTER, 
                         apply_enhancement=True, stop_check_func=None) -> str:
    """
    Convenience function that combines smart recording with transcription.
    Uses in-memory processing for better performance.
    
    Args:
        sample_rate: Audio sample rate for recording
        max_wait: Maximum recording time
        silence_limit: Silence duration before stopping
        threshold: Speech detection threshold
        language: Language for transcription
        vad_filter: Voice activity detection for transcription
        apply_enhancement: Whether to apply voice enhancement
        stop_check_func: Function to check if manual stop was requested (returns bool)
    
    Returns:
        str: Transcribed text, or None if failed
    """
    print("🎙️ Starting smart record and transcribe with in-memory processing...")
    
    # Record audio smartly (this will save a file for now, but we'll load it back into memory)
    audio_path = record_audio_smart(
        sample_rate=sample_rate,
        max_wait=max_wait,
        silence_limit=silence_limit,
        threshold=threshold,
        apply_enhancement=False,  # We'll apply enhancement during transcription
        stop_check_func=stop_check_func
    )
    
    if not audio_path:
        print("❌ Recording failed, cannot transcribe")
        return None
    
    try:
        # Load the recorded audio into memory
        audio_data, recorded_sr = load_audio_any(audio_path, target_sr=sample_rate)
        
        # Clean up the temporary recording file
        try:
            os.remove(audio_path)
            print(f"🗑️  Cleaned up temporary recording file")
        except Exception as e:
            print(f"⚠️  Could not remove temporary file: {e}")
        
        # Transcribe using in-memory processing
        print("🔄 Transcribing recorded audio with in-memory processing...")
        transcription = transcribe_audio_data(
            audio_data, recorded_sr, language, vad_filter, apply_enhancement
        )
        
        return transcription
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None
