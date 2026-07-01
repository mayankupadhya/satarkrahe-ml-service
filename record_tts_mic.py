import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
import os
import time

SAMPLE_RATE = 16000
DURATION = 6  # 6 seconds per recording

tts_files = sorted([f for f in os.listdir("tts_mic_fake") if f.endswith(".mp3")])
os.makedirs("final_dataset_v5/train/fake", exist_ok=True)

print(f"Found {len(tts_files)} TTS files to record")
print("Apna speaker volume thoda upar rakho, mic ke paas rakho")
print("3 second baad shuru hoga...\n")
time.sleep(3)

for i, tts_file in enumerate(tts_files):
    print(f"Recording {i+1}/{len(tts_files)}: {tts_file}")

    # Convert MP3 to WAV for playback
    mp3_path = f"tts_mic_fake/{tts_file}"
    wav_path = f"tts_mic_fake/{tts_file.replace('.mp3', '_play.wav')}"
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "44100", "-ac", "2", wav_path
    ], capture_output=True)

    # Start recording from mic
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    # Play the TTS file through speaker
    play_data, play_sr = sf.read(wav_path)
    sd.play(play_data, play_sr)

    # Wait for recording to finish
    sd.wait()

    # Save recorded audio to fake dataset
    output_path = f"final_dataset_v5/train/fake/mic_tts_{i+1:02d}.wav"
    sf.write(output_path, recording, SAMPLE_RATE)
    print(f"  Saved: {output_path}")

    time.sleep(0.5)

print(f"\nDone! {len(tts_files)} mic-recorded TTS samples fake dataset mein add ho gaye.")
