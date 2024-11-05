import libreria_tmp as tmp
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import torch
from opts import parse_opts
from model import generate_model
if __name__ == '__main__':
    emotion_dict=['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []

    model, _ = generate_model(opt)
    model.eval()
    model.to(opt.device)
        
    # Questo pth è il pretreinato da GiuseppeF 
    best_state = torch.load('/Users/renatoesposito/Desktop/cognitive-robotics-project/modello-test/results/RAVDESS_multimodalcnn_15_best0.pth', map_location=torch.device('cpu'))
    #Questo è il nostro 
    #best_state = torch.load('/Users/renatoesposito/Desktop/cognitive-robotics-project/end-to-end-mm-ers/models/RAVDESS_multimodalcnn_15_best0_04_11.pth', map_location=torch.device('cpu'))
    model.load_state_dict(best_state['state_dict'])
    print("Tutto ok")
    input_path="/Users/renatoesposito/Desktop/cognitive-robotics-project/end-to-end-mm-ers/raw_data/angry_wrong_1.mp4"
    print("Passo alla funzione")
    audio_var, video_var = tmp.predict_single_video(input_path,video_norm_value=opt.video_norm_value, batch_size=opt.batch_size)
    with torch.no_grad():
        output = model(x_audio=audio_var, x_visual=video_var)
    print("Output: ", output)
    result = output.argmax(1)
    print("Prediction: ",emotion_dict[result])
    
