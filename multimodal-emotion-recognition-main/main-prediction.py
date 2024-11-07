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

    #PROBLEMA: SU MAC vuole cuda e il modello non funziona senza cuda questa cosa va sistemata forse nel trainig. 
    best_state = torch.load('c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/lt_1head_moddrop_2.pth')
    model.load_state_dict(best_state['state_dict'])
    
    input_path="c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/raw_data/angry_wrong_1.mp4"
    audio_var, video_var = preprocessing.predict_single_video(input_path,video_norm_value=opt.video_norm_value, batch_size=opt.batch_size)
    with torch.no_grad():
        output = model(x_audio=audio_var, x_visual=video_var)
    print("[LOGITS] Output: ", output)
    # Calcola le probabilità con softmax
    '''probabilities = F.softmax(output, dim=1)
    percentages = probabilities*100
    percentages = percentages.detach().numpy().flatten()
    # Converti in percentuali
    result = output.argmax(1)
    for i, p in enumerate(percentages):
        print(f"Classe {emotion_dict[i]}: {p:.4f}%")
    print("Prediction: ",emotion_dict[result])'''
    
