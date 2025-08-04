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
INPUT_PATH       = r"send_500_rs_to_hnb.mp3"
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

ENHANCED_OUTPUT_DIR = r"C:\Projects\voice_agent\voice_enhancement"
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


def record_audio_smart(sample_rate=RECORDING_SAMPLE_RATE, max_wait=MAX_WAIT, 
                      silence_limit=SILENCE_LIMIT, threshold=THRESHOLD,
                      output_dir=None, apply_enhancement=True):
    """
    Smart audio recording that automatically stops when user stops speaking.
    Applies voice enhancement in memory and returns the path to the saved audio file.
    
    Args:
        sample_rate: Audio sample rate (Hz)
        max_wait: Maximum total recording time (seconds)
        silence_limit: How long to wait after silence before stopping (seconds)
        threshold: RMS threshold for detecting speech
        output_dir: Directory to save the recording (defaults to voice_enhancement dir)
        apply_enhancement: Whether to apply voice enhancement in memory
    
    Returns:
        str: Path to the saved audio file, or None if recording failed
    """
    if output_dir is None:
        output_dir = ENHANCED_OUTPUT_DIR
    
    # Prepare recording variables
    recorded_audio = []
    start_time = time.time()
    silent_start = None
    stop_flag = False
    block_size = int(BLOCK_DURATION * sample_rate)
    
    print(f"🎤 Smart recording started. Speak now...")
    print(f"📊 Settings: Max wait={max_wait}s, Silence limit={silence_limit}s, Threshold={threshold}")
    if apply_enhancement:
        print(f"🔧 Voice enhancement will be applied in memory")
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal silent_start, stop_flag, recorded_audio
        
        if status:
            print(f"InputStream Status: {status}")
        
        # Store the audio data
        recorded_audio.append(indata.copy())
        
        # Use first channel for mono analysis
        audio_block = indata[:, 0] if indata.ndim > 1 else indata
        rms = np.sqrt(np.mean(audio_block**2))
        
        now = time.time()
        
        # Check if we're detecting speech
        if rms > threshold:
            if silent_start is not None:
                print(f"🗣️  Speech detected (RMS: {rms:.4f})")
            silent_start = None  # Reset silence timer on speech
        else:
            if silent_start is None:
                silent_start = now  # Mark silence start
                print(f"🤫 Silence started (RMS: {rms:.4f})")
        
        # Calculate time since recording started and since silence began
        since_start = now - start_time
        since_silence = (now - silent_start) if silent_start else 0.0
        
        # Show live status every few blocks
        if len(recorded_audio) % 20 == 0:  # Every ~2 seconds
            if silent_start:
                print(f"⏱️  Recording: {since_start:.1f}s | Silence: {since_silence:.1f}s/{silence_limit}s")
            else:
                print(f"⏱️  Recording: {since_start:.1f}s | Speaking...")
        
        # Stop criteria
        if since_start >= max_wait:
            print(f"⏰ Maximum recording time ({max_wait}s) reached")
            stop_flag = True
            raise sd.CallbackStop()
        elif silent_start and since_silence >= silence_limit:
            print(f"🛑 Stopping after {silence_limit}s of silence")
            stop_flag = True
            raise sd.CallbackStop()
    
    try:
        # Open the microphone stream
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=block_size,
            callback=audio_callback,
            dtype=np.float32
        ):
            # Keep the main thread alive while recording
            while not stop_flag:
                time.sleep(0.1)
        
        print("⏹️  Recording stopped.")
        
        # Process the recorded audio
        if not recorded_audio:
            print("❌ No audio was recorded")
            return None
        
        # Concatenate all audio blocks
        full_audio = np.concatenate(recorded_audio, axis=0)
        if full_audio.ndim > 1:
            full_audio = full_audio[:, 0]  # Convert to mono
        
        # Normalize audio
        if np.max(np.abs(full_audio)) > 0:
            full_audio = full_audio / np.max(np.abs(full_audio)) * 0.95
        
        # Apply voice enhancement in memory if requested
        if apply_enhancement:
            print("🔧 Applying voice enhancement to recorded audio...")
            enhanced_audio, enhanced_sr = voice_enhancement_in_memory(full_audio, sample_rate)
            final_audio = enhanced_audio
            final_sr = enhanced_sr
        else:
            final_audio = full_audio
            final_sr = sample_rate
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "smart_enhanced" if apply_enhancement else "smart_recording"
        output_filename = f"{prefix}_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the audio file
        sf.write(output_path, final_audio, final_sr, subtype="PCM_16")
        
        duration = len(final_audio) / final_sr
        print(f"✅ Smart recording saved: {output_path}")
        print(f"📊 Duration: {duration:.2f}s, Samples: {len(final_audio)}")
        if apply_enhancement:
            print(f"🔧 Voice enhancement applied successfully")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Smart recording failed: {e}")
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
                         apply_enhancement=True) -> str:
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
        apply_enhancement=False  # We'll apply enhancement during transcription
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


def record_and_transcribe_pure_memory(sample_rate=RECORDING_SAMPLE_RATE, max_wait=MAX_WAIT,
                                     silence_limit=SILENCE_LIMIT, threshold=THRESHOLD,
                                     language=LANGUAGE, vad_filter=VAD_FILTER, 
                                     apply_enhancement=True) -> str:
    """
    Pure in-memory version that doesn't save any files at all.
    Records directly to memory and processes without temporary files.
    
    Args:
        sample_rate: Audio sample rate for recording
        max_wait: Maximum recording time
        silence_limit: Silence duration before stopping
        threshold: Speech detection threshold
        language: Language for transcription
        vad_filter: Voice activity detection for transcription
        apply_enhancement: Whether to apply voice enhancement
    
    Returns:
        str: Transcribed text, or None if failed
    """
    print("🎙️ Starting pure in-memory record and transcribe...")
    
    # Prepare recording variables
    recorded_audio = []
    start_time = time.time()
    silent_start = None
    stop_flag = False
    block_size = int(BLOCK_DURATION * sample_rate)
    
    print(f"📊 Settings: Max wait={max_wait}s, Silence limit={silence_limit}s, Threshold={threshold}")
    if apply_enhancement:
        print(f"🔧 Voice enhancement will be applied in memory")
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal silent_start, stop_flag, recorded_audio
        
        if status:
            print(f"InputStream Status: {status}")
        
        # Store the audio data
        recorded_audio.append(indata.copy())
        
        # Use first channel for mono analysis
        audio_block = indata[:, 0] if indata.ndim > 1 else indata
        rms = np.sqrt(np.mean(audio_block**2))
        
        now = time.time()
        
        # Check if we're detecting speech
        if rms > threshold:
            if silent_start is not None:
                print(f"🗣️  Speech detected (RMS: {rms:.4f})")
            silent_start = None  # Reset silence timer on speech
        else:
            if silent_start is None:
                silent_start = now  # Mark silence start
                print(f"🤫 Silence started (RMS: {rms:.4f})")
        
        # Calculate time since recording started and since silence began
        since_start = now - start_time
        since_silence = (now - silent_start) if silent_start else 0.0
        
        # Show live status every few blocks
        if len(recorded_audio) % 20 == 0:  # Every ~2 seconds
            if silent_start:
                print(f"⏱️  Recording: {since_start:.1f}s | Silence: {since_silence:.1f}s/{silence_limit}s")
            else:
                print(f"⏱️  Recording: {since_start:.1f}s | Speaking...")
        
        # Stop criteria
        if since_start >= max_wait:
            print(f"⏰ Maximum recording time ({max_wait}s) reached")
            stop_flag = True
            raise sd.CallbackStop()
        elif silent_start and since_silence >= silence_limit:
            print(f"🛑 Stopping after {silence_limit}s of silence")
            stop_flag = True
            raise sd.CallbackStop()
    
    try:
        # Record audio
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=block_size,
            callback=audio_callback,
            dtype=np.float32
        ):
            while not stop_flag:
                time.sleep(0.1)
        
        print("⏹️  Recording stopped.")
        
        # Process the recorded audio
        if not recorded_audio:
            print("❌ No audio was recorded")
            return None
        
        # Concatenate all audio blocks
        full_audio = np.concatenate(recorded_audio, axis=0)
        if full_audio.ndim > 1:
            full_audio = full_audio[:, 0]  # Convert to mono
        
        # Normalize audio
        if np.max(np.abs(full_audio)) > 0:
            full_audio = full_audio / np.max(np.abs(full_audio)) * 0.95
        
        duration = len(full_audio) / sample_rate
        print(f"✅ Recorded {duration:.2f}s of audio ({len(full_audio)} samples)")
        
        # Transcribe directly from memory
        print("🔄 Transcribing audio directly from memory...")
        transcription = transcribe_audio_data(
            full_audio, sample_rate, language, vad_filter, apply_enhancement
        )
        
        return transcription
        
    except Exception as e:
        print(f"❌ Pure memory record and transcribe failed: {e}")
        return None
