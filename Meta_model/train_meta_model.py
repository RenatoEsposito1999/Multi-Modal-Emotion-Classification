import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.generate_dataset_RAVDESS import get_training_set_RAVDESS
from datasets.seediv_dataset import generate_dataset_SEEDIV
from utils import transforms
import torch
from torch.utils.data import DataLoader


def train_meta_classifier(opts, stacking_classifier):
   
    # Set up video transforms
    video_transform = transforms.Compose([
        transforms.ToTensor(opts.video_norm_value)
    ])

    #Load dataset eeg
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)
    
    
    av_dataset = get_training_set_RAVDESS(opts, spatial_transform=video_transform)
    

    # Create DataLoaders with proper batch size
    train_loader_av = DataLoader(
        av_dataset, 
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.n_threads,
        pin_memory=True
    )
    
    train_loader_eeg = DataLoader(
        eeg_dataset, 
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.n_threads,
        pin_memory=True
    )

    print("Starting training...")
    stacking_classifier.fit(
        train_loader_av,
        train_loader_eeg,
        epochs=100,
        patience=5
    )

    torch.save({
            'model_av_state_dict': stacking_classifier.model1.state_dict(),
            'model_eeg_state_dict': stacking_classifier.model2.state_dict(),
            'meta_model_state_dict': stacking_classifier.meta_model
            }, 'results/Complete_model.pth')

    