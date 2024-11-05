from video_preprocessing import Video_preprocessing
from audio_preprocessing import Audio_preprocessing

video = Video_preprocessing("/Users/renatoesposito/Desktop/cognitive-robotics-project/end-to-end-mm-ers/raw_data/angry_wrong_1.mp4")
video_npy = video.process()

audio = Audio_preprocessing("/Users/renatoesposito/Desktop/cognitive-robotics-project/end-to-end-mm-ers/raw_data/angry_wrong_1.mp4")
audio_npy = audio.process()