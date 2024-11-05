from model import get_model
from extract import *
import torch

emotion_dict=['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
path = 'C:/Users/Vince/Desktop/End-to-End-Multimodal-emotion-recognition/raw_data/angry_wrong_1.mp4'
if __name__ == '__main__':
    # load the trained model
    model = get_model()
    # load visual data and audio data from raw video file
    inputs_visual,inputs_audio= extract(path)
    # predict
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_audio, inputs_visual)
    print("Output: ", outputs)
    result = outputs.argmax(1)
    print(emotion_dict[result])