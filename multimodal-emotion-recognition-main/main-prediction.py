import input_preprocessing as preprocessing
import torch.nn.functional as F
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
    input_path="/Users/renatoesposito/Desktop/cognitive-robotics-project/modello-test/raw_data/angry_wrong_1.mp4"
    audio_var, video_var = preprocessing.predict_single_video(input_path,video_norm_value=opt.video_norm_value, batch_size=opt.batch_size)
    with torch.no_grad():
        output = model(x_audio=audio_var, x_visual=video_var)
    print("[LOGITS] Output: ", output)
    # Calcola le probabilità con softmax
    probabilities = F.softmax(output, dim=1)
    percentages = probabilities*100
    percentages = percentages.detach().numpy().flatten()
    # Converti in percentuali
    result = output.argmax(1)
    for i, p in enumerate(percentages):
        print(f"Classe {emotion_dict[i]}: {p:.4f}%")
    print("Prediction: ",emotion_dict[result])
    
