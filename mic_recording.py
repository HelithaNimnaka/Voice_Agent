import sounddevice as sd
import numpy as np
import wave
import sys
import datetime

def record_audio(filename, duration=15, samplerate=44100):
    print(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    
    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    # Determine output filename: use arg if provided, else timestamp-based default
    if len(sys.argv) == 1:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"recording_{ts}.wav"
        print(f"No filename provided, using default '{output_filename}'")
    elif len(sys.argv) == 2:
        output_filename = sys.argv[1]
    else:
        print("Usage: python voice_recording.py [output_filename]")
        sys.exit(1)
    record_audio(output_filename)