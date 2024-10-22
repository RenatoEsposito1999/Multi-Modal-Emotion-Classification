from video_preprocessing import Video_preprocessing
from audio_preprocessing import Audio_preprocessing

video = Video_preprocessing("C:/Users/Vince/Desktop/cognitive-robotics-project/test_vincenzo/prova_3.mp4")
video_npy = video.process()

audio = Audio_preprocessing("C:/Users/Vince/Desktop/cognitive-robotics-project/test_vincenzo/prova_3.mp4")
audio_npy = audio.process()