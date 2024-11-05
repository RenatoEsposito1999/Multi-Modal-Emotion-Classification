import librosa
import os
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip

class Audio_preprocessing():
    def __init__ (self, video_path):
        self.video_path = video_path
        self.target_time = 3.6
    
    def process(self):
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile("prova.wav")
        audios = librosa.core.load("./prova.wav", sr=22050)

        y = audios[0]
        sr = audios[1]
        target_length = int(sr * self.target_time)
        if len(y) < target_length:
            y = np.array(list(y) + [0 for i in range(target_length - len(y))])
        else:
            remain = len(y) - target_length
            y = y[remain // 2:-(remain - remain // 2)]

        sf.write("prova_croppad.wav", y, sr)
        return y