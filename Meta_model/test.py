import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')
sys.path.append('../Shared')

from datasets.seediv_dataset import generate_dataset_SEEDIV
import torch
from torch.utils.data import DataLoader
from datasets.generate_dataset_RAVDESS import get_test_set_RAVDESS
from utils import transforms
from plot_data import *


def testing(opts, stacking_classifier):
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)

    # Load weights
    weights = torch.load('results/Complete_model.pth')
    stacking_classifier.meta_model = weights['meta_model_state_dict']
    stacking_classifier.eval()
    
     # Set up video transforms
    video_transform = transforms.Compose([
        transforms.ToTensor(opts.video_norm_value)
    ])
    
    av_dataset = get_test_set_RAVDESS(opts, spatial_transform=video_transform)
    
     # Create DataLoaders with proper batch size
    test_loader_av = DataLoader(
        av_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=opts.n_threads,
        pin_memory=True
    )
    
    test_loader_eeg = DataLoader(
        eeg_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=opts.n_threads,
        pin_memory=True
    )
    
    final_predictions, targets = stacking_classifier.test(test_loader_av, test_loader_eeg)
    
    compute_confusion_matrix(targets, final_predictions, "./Images/confusion_matrix.jpeg")
    

    
    
    
    