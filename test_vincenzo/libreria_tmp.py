import torch
import cv2
import numpy as np
from moviepy.editor import AudioFileClip
import torchaudio
from torchvision import transforms
import transforms
import functools
from PIL import Image
import librosa
from torch.autograd import Variable


'''
input_size=(224, 224)
frame = cv2.resize(frame, input_size)
video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(video_norm_value)])

def extract_audio_features(audio_path, sample_rate=16000, num_channels=10):
    waveform, sr = torchaudio.load(audio_path)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    waveform = transform(waveform)

    # Convert stereo to mono by averaging channels (using torch.mean)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Adjust number of channels (replicate channels if necessary)
    if waveform.size(0) < num_channels:
        waveform = waveform.repeat(num_channels // waveform.size(0), 1)

    return waveform.unsqueeze(0)  # Add batch dimension
'''
def load_audio(audiofile, sr=22050):
    y, sr = librosa.core.load(audiofile, sr=sr)
    return y, sr

def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    #video = video_dir_path   
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def preprocess_frame(frame, input_size=(224, 224), video_norm_value=None):
    '''# Convert BGR (OpenCV) to RGB
    
    # Ridimensiona l'immagine
    
    # Trasformazioni (puoi aggiungere altre se necessarie)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte in tensor [C, H, W]
    ])'''
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, input_size)
    video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(video_norm_value)])
    
    return video_transform(frame)

def predict_single_video(video_npy, audio_path, model, input_size=(224, 224), device='cpu', frames_per_sample=15, video_norm_value=None, batch_size=1):
    model.eval()
    model.to(device)
    loader = get_default_video_loader()
    visual_input_batch = loader(video_npy)
    #VIDEO
    video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(video_norm_value)])
    video_transform.randomize_parameters()
    clip = [video_transform(img) for img in visual_input_batch]            
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    #AUDIO
    y, sr = load_audio(audio_path, sr=22050) 
    mfcc = get_mfccs(y, sr)            
    audio_features = mfcc
    audio_var = torch.from_numpy(audio_features)
    print(f"Dimensioni dell'audio prima di essere processato: {audio_var.shape}")
    print(f"Dimensioni del video prima di essere processato: {clip.shape}")
    '''
        Dimensioni dell'audio prima di essere processato: torch.Size([10, 156])
        Dimensioni del video prima di essere processato: torch.Size([3, 15, 224, 224])
    '''
    clip = clip.unsqueeze(0).expand(batch_size, -1, -1,-1, -1)
    audio_var = audio_var.unsqueeze(0).expand(batch_size, -1, -1)
    
    
    with torch.no_grad():
        video_var = Variable(clip)
        audio_var = Variable(audio_var)
    video_var = video_var.permute(0,2,1,3,4)
    video_var = video_var.reshape(video_var.shape[0]*video_var.shape[1], video_var.shape[2], video_var.shape[3], video_var.shape[4])
    print(f"Dimensioni dell'audio preprocessato: {audio_var.shape}")
    print(f"Dimensioni del video preprocessato: {video_var.shape}")
    logits = model(x_audio=audio_var, x_visual=video_var)
    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    print(predicted_classes)
    
    
    '''# Estrarre l'audio dal video
    video_clip = AudioFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.write_audiofile(audio_path, codec='pcm_s16le')
    audio_input = extract_audio_features(audio_path).to(device)
'''
    '''# Apertura del video
    cap = cv2.VideoCapture(video_path)
    
    predictions = []
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        visual_input = preprocess_frame(frame, input_size, video_norm_value=video_norm_value).to(device)
        frame_buffer.append(visual_input)

        if len(frame_buffer) == frames_per_sample:
            visual_input_batch = torch.stack(frame_buffer).to(device)
            frame_buffer = []'''

    '''with torch.no_grad():
        logits = model(x_visual=visual_input_batch, x_audio=audio_input)
                
        # Stampa i logits per ispezionarli
        print("Logits:", logits)
                
        # Applica softmax e stampa le probabilità
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        print("Probabilità:", probabilities)
                
        predictions.append(probabilities.cpu().numpy())

    if 0 < len(frame_buffer) < frames_per_sample:
        last_frame = frame_buffer[-1]
        while len(frame_buffer) < frames_per_sample:
            frame_buffer.append(last_frame)

        visual_input_batch = torch.stack(frame_buffer).to(device)
        
        with torch.no_grad():
            logits = model(x_visual=visual_input_batch, x_audio=audio_input)
            # Ottieni i logits dal modello

            # Calcola le probabilità (opzionale, se vuoi vedere anche le probabilità)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Trova l'indice della classe con la probabilità più alta
            predicted_class = torch.argmax(logits, dim=-1)

            # Stampa i risultati
            #print(f"Logits: {logits}")
            #print(f"Probabilità: {probabilities}")
            print(f"Classe predetta: {predicted_class.item()}")  # .item() se è un singolo valore

    cap.release()
    
    return np.array(predictions)'''
    