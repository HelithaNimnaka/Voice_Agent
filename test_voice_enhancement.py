import os
import time
import torch
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime

from faster_whisper import WhisperModel

# === CONFIG (edit these manually) ===
INPUT_PATH = r"C:\Projects\voice_agent\received_audio_20250630_122220.wav"
MODEL_SIZE = "tiny"            # choices: tiny, base, small, medium, large
COMPUTE_TYPE = "float32"       # choices: float16, int8, float32
LANGUAGE = "en"                # or None for autodetect
ENABLE_ENHANCEMENT = True      # set False to skip enhancement entirely
VAD_FILTER = True              # voice activity detection filtering
# ====================================

# Fixed enhancement output directory
ENHANCED_OUTPUT_DIR = r"C:\Projects\voice_agent\voice_enhancement"
os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)

# ClearVoice import (may raise if not installed)
try:
    from clearvoice import ClearVoice
    HAVE_CLEARVOICE = True
except ImportError:
    HAVE_CLEARVOICE = False
    print("Warning: ClearVoice not installed; enhancement will be skipped.")

# Global whisper model holder
_whisper_model = None

def init_whisper(model_size: str, device: str, compute_type: str):
    global _whisper_model
    if _whisper_model is None or (
        getattr(_whisper_model, "model_size", None) != model_size
        or getattr(_whisper_model, "device", None) != device
        or getattr(_whisper_model, "compute_type", None) != compute_type
    ):
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_model.model_size = model_size
        _whisper_model.device = device
        _whisper_model.compute_type = compute_type
    return _whisper_model

def init_clearvoice():
    if not HAVE_CLEARVOICE:
        return None
    return ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

_myClearVoice = init_clearvoice()

def load_audio_any(path: str, target_sr: int = 48000):
    """
    Robust audio loader: try librosa first, then fall back to pydub.
    Returns (y, sr) as float32 mono.
    """
    # Try librosa (which uses soundfile / audioread)
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype("float32"), sr
    except Exception as e:
        print(f"[audio load] librosa failed ({e}), trying pydub fallback.")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            # Normalize from integer range depending on sample width
            max_int = float(1 << (8 * audio.sample_width - 1))
            y = samples / max_int
            return y.astype("float32"), target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio via both librosa and pydub: {e2}")

# Fallback spectral gating denoiser
def _spectral_gate(y: np.ndarray, sr: int, n_fft=1024, hop_length=256, prop_decrease=1.0) -> np.ndarray:
    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S_full), np.angle(S_full)
    frame_energies = np.sum(magnitude, axis=0)
    thresh = np.percentile(frame_energies, 10)
    noise_frames = magnitude[:, frame_energies <= thresh]
    if noise_frames.size == 0:
        noise_profile = np.mean(magnitude, axis=1, keepdims=True)
    else:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    mask_gain = (magnitude - noise_profile) / (magnitude + 1e-8)
    mask_gain = np.clip(mask_gain, 0, 1) ** prop_decrease
    enhanced_mag = magnitude * mask_gain
    S_enhanced = enhanced_mag * np.exp(1j * phase)
    y_enhanced = librosa.istft(S_enhanced, hop_length=hop_length)
    max_val = np.max(np.abs(y_enhanced)) + 1e-9
    y_enhanced = y_enhanced / max_val * 0.99
    return y_enhanced

def enhance_audio_fallback(input_path: str) -> str:
    # Load audio (target 48k)
    y, sr = load_audio_any(input_path, target_sr=48000)
    enhanced = None
    try:
        import noisereduce as nr
        noise_sample_len = min(len(y), int(0.5 * sr))
        noise_sample = y[:noise_sample_len]
        enhanced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
        print("[enhancement-fallback] Used noisereduce")
    except ImportError:
        enhanced = _spectral_gate(y, sr, prop_decrease=1.0)
        print("[enhancement-fallback] Used spectral gating fallback")

    base = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enhanced_filename = f"{base}_enhanced_fallback_{ts}.wav"
    enhanced_path = os.path.join(ENHANCED_OUTPUT_DIR, enhanced_filename)
    sf.write(enhanced_path, enhanced.astype("float32"), sr, subtype="PCM_16")
    print(f"[enhancement-fallback] Saved fallback enhanced audio to: {enhanced_path}")
    return enhanced_path

def voice_enhancement(filename: str) -> str:
    """
    Try ClearVoice; if it fails or returns invalid output, fall back to local denoising.
    """
    if not ENABLE_ENHANCEMENT:
        return filename

    if _myClearVoice:
        try:
            enhanced_audio = _myClearVoice(input_path=filename, online_write=False)
            if enhanced_audio is None or not hasattr(enhanced_audio, "ndim"):
                raise ValueError("ClearVoice returned no usable audio")
            # Squeeze / pick channel
            if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
                audio_to_save = enhanced_audio.squeeze(0)
            elif enhanced_audio.ndim == 2 and enhanced_audio.shape[1] == 1:
                audio_to_save = enhanced_audio.squeeze(1)
            elif enhanced_audio.ndim == 1:
                audio_to_save = enhanced_audio
            else:
                audio_to_save = enhanced_audio[0] if enhanced_audio.ndim == 2 else enhanced_audio

            if audio_to_save.dtype != "float32":
                audio_to_save = audio_to_save.astype("float32")

            peak = max(abs(audio_to_save.max()), abs(audio_to_save.min()))
            if peak > 1.0:
                audio_to_save = audio_to_save / peak * 0.99

            base = os.path.splitext(os.path.basename(filename))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            enhanced_filename = f"{base}_enhanced_clearvoice_{ts}.wav"
            enhanced_path = os.path.join(ENHANCED_OUTPUT_DIR, enhanced_filename)
            print(f"[enhancement] Saving ClearVoice enhanced audio to: {enhanced_path}")
            sf.write(enhanced_path, audio_to_save, 48000, subtype="PCM_16")
            return enhanced_path
        except Exception as e:
            print(f"[enhancement] ClearVoice failed ({e}), falling back to local enhancement.")

    return enhance_audio_fallback(filename)

def transcribe(audio_path: str, model_size: str = "medium", compute_type: str = "float16", language: str = "en", vad_filter: bool = True) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_whisper(model_size, device, compute_type)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=vad_filter,
    )
    full_text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "full_text": full_text,
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segments
        ],
        "language": info.language,
        "duration": info.duration,
    }

def main():
    if not os.path.isfile(INPUT_PATH):
        print(f"Input file does not exist: {INPUT_PATH}")
        return

    print(f"Input audio: {INPUT_PATH}")
    start_total = time.time()

    if ENABLE_ENHANCEMENT:
        processed_path = voice_enhancement(INPUT_PATH)
    else:
        processed_path = INPUT_PATH

    print(f"[transcribe] Running transcription (model={MODEL_SIZE}, compute_type={COMPUTE_TYPE}) on: {processed_path}")
    t0 = time.time()
    try:
        result = transcribe(
            processed_path,
            model_size=MODEL_SIZE,
            compute_type=COMPUTE_TYPE,
            language=LANGUAGE,
            vad_filter=VAD_FILTER,
        )
    except Exception as e:
        print(f"[transcribe] Transcription failed: {e}")
        return
    t1 = time.time()

    print("\n=== Final Transcription ===")
    print(f"Detected language: {result.get('language')}")
    print(f"Full text: {result.get('full_text')}\n")
    print("Segments:")
    for seg in result["segments"]:
        print(f"  [{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
    print(f"\nTranscription time: {t1 - t0:.2f}s")
    print(f"Total elapsed: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()
