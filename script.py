import cv2
import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
import imageio_ffmpeg as iio
import subprocess

# Parametri di registrazione
duration = 10  # Durata in secondi
fps = 30  # Frame rate del video
output_video = 'output.mp4'
temp_video = 'temp_video.avi'
temp_audio = 'output_audio.wav'

frames = []
timestamps = []
audio_recording = []

# Funzione per registrare video
def record_video():
    global frames, timestamps
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        timestamps.append(time.perf_counter() - start_time)
        time.sleep(1 / fps)
    
    cap.release()

# Funzione per registrare audio
def record_audio():
    global audio_recording
    audio_recording = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
    sd.wait()

# Thread per registrare audio e video simultaneamente
video_thread = threading.Thread(target=record_video)
audio_thread = threading.Thread(target=record_audio)

video_thread.start()
audio_thread.start()

video_thread.join()
audio_thread.join()

# Salva il video
height, width, _ = frames[0].shape
out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

for frame in frames:
    out.write(frame)

out.release()

# Salva l'audio
write(temp_audio, 44100, audio_recording)

# Allinea audio e video con ffmpeg
subprocess.run([
    iio.get_ffmpeg_exe(),
    '-i', temp_video,
    '-i', temp_audio,
    '-c:v', 'libx264',
    '-c:a', 'aac',
    '-strict', 'experimental',
    output_video
], check=True)

print(f"Registrazione completata. File salvato come {output_video}.")
