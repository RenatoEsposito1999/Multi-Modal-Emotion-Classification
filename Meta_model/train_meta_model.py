import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.generate_dataset_RAVDESS import get_training_set_RAVDESS
from opts_meta_model import parse_opts 
from torcheeg.models import FBCCNN
from datasets.seediv_dataset import generate_dataset_SEEDIV
from Multimodal_transformer.MultimodalTransformer import MultimodalTransformer
from EmotionStackingClassigier import EmotionStackingClassifier
from utils import transforms
#from Data_preprocessing import input_preprocessing_predict
import random

import torch
import numpy as np
from torch import nn
#from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
#from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_meta_classifier():
    opts = parse_opts()
    
    # Set up video transforms
    video_transform = transforms.Compose([
        transforms.ToTensor(opts.video_norm_value)
    ])

    # Load models and datasets
    model_eeg = FBCCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(opts.device)
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)
    
    
    av_dataset = get_training_set_RAVDESS(opts, spatial_transform=video_transform)
    
    # Initialize AV model
    model_av = MultimodalTransformer(
        opts.n_classes, 
        seq_length=opts.sample_duration,
        pretr_ef=opts.pretrain_path,
        num_heads=opts.num_heads
    )

    # Move AV model to appropriate device
    if opts.device != 'cpu':
        model_av = model_av.to(opts.device)
        model_av = nn.DataParallel(model_av, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model_av.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

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

    # Load model weights
    model_eeg.load_state_dict(
        torch.load('../EEG_model/results/best_state.pth')['state_dict']
    )
    model_eeg.to(opts.device)
    
    # Load AV model weights based on device
    av_weights_path = f'../audio_video_emotion_recognition_model/{opts.result_path}/{opts.store_name}_best{"_cpu_" if opts.device == "cpu" else ""}.pth'
    av_state = torch.load(av_weights_path, map_location=opts.device)
    model_av.load_state_dict(av_state['state_dict'])

    # Set models to eval mode
    model_av.eval()
    model_eeg.eval()

    # Initialize and train stacking classifier
    print("Initializing stacking classifier...")
    stacking_classifier = EmotionStackingClassifier(
        model1=model_av,
        model2=model_eeg,
        opts=opts,
        max_iter=1000
    )

    print("Starting training...")
    stacking_classifier.fit(
        train_loader_av,
        train_loader_eeg,
        epochs=100,
        patience=5
    )

    torch.save({
            'model_av_state_dict': model_av.state_dict(),
            'model_eeg_state_dict': model_eeg.state_dict(),
            'meta_model_state_dict': stacking_classifier.meta_model
            }, 'results/Complete_model.pth')
    
    return stacking_classifier