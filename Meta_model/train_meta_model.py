import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.generate_dataset_RAVDESS import get_training_set_RAVDESS, get_validation_set_RAVDESS, get_test_set_RAVDESS
from datasets.seediv_dataset import generate_dataset_SEEDIV
from utils import transforms
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

def prepare_dataset(training, validation, perc_train, perc_val):
    # Calcola la dimensione totale desiderata
    total_size = len(training) * perc_train + len(validation) * perc_val

    # Calcola le dimensioni effettive per rispettare la proporzione
    train_size = int(total_size * perc_train)  # 30% del totale desiderato
    val_size = int(total_size * perc_val)    # 70% del totale desiderato

    # Assicurati di non eccedere le dimensioni disponibili
    train_size = min(train_size, len(training))
    val_size = min(val_size, len(validation))
    
    subset_train, _ = random_split(training, [train_size, len(training) - train_size])
    subset_val, _ = random_split(validation, [val_size, len(validation) - val_size])
    dataset = ConcatDataset([subset_train, subset_val])
    
    return dataset
    

def train_meta_classifier(opts, stacking_classifier):
   
    # Set up video transforms
    video_transform = transforms.Compose([
        transforms.ToTensor(opts.video_norm_value)
    ])

    #Load dataset eeg
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)
    
    num_training = int(len(eeg_dataset) * 0.8)
    num_val = int(len(eeg_dataset) * 0.1)
    num_test = len(eeg_dataset) - num_val - num_training

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, _ = random_split(eeg_dataset, [num_training, num_val, num_test], generator=generator)
    
    eeg_dataset = prepare_dataset(train_dataset, val_dataset, 0.2, 0.8)
    
    
    av_dataset_training = get_training_set_RAVDESS(opts, spatial_transform=video_transform)
    av_dataset_validation = get_validation_set_RAVDESS(opts, spatial_transform=video_transform)
    
    av_dataset = prepare_dataset(av_dataset_training, av_dataset_validation, 0.2, 0.8)
    
           
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
            'meta_model_state_dict': stacking_classifier.meta_model
            }, 'results/Complete_model.pth')

    