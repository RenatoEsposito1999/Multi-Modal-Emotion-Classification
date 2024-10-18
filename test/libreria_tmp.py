import torch
import cv2
import numpy as np
from moviepy.editor import AudioFileClip
import torchaudio
from torchvision import transforms

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

def preprocess_frame(frame, input_size=(224, 224)):
    # Convert BGR (OpenCV) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Ridimensiona l'immagine
    frame = cv2.resize(frame, input_size)
    # Trasformazioni (puoi aggiungere altre se necessarie)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte in tensor [C, H, W]
    ])
    return transform(frame).unsqueeze(0)  # Aggiunge dimensione per il batch

def predict_single_video(video_path, model, input_size=(224, 224), device='cpu'):
    model.eval()
    model.to(device)
    
    # Estrarre l'audio direttamente dal video
    video_clip = AudioFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.write_audiofile(audio_path, codec='pcm_s16le')
    audio_input = extract_audio_features(audio_path).to(device)

    # Apertura del video
    cap = cv2.VideoCapture(video_path)
    
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessamento del frame
        visual_input = preprocess_frame(frame, input_size).to(device)
        visual_input = visual_input.unsqueeze(0)  # Aggiungi dimensione batch
        # Passa l'audio e il video al modello
        with torch.no_grad():
            output = model(x_visual=visual_input, x_audio=audio_input)

        predictions.append(output.cpu().numpy())  # Memorizza i risultati
        
    cap.release()
    
    return np.array(predictions)
