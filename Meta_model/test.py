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
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader,random_split
import csv


def testing(opts, stacking_classifier):
    eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)
    
    num_training = int(len(eeg_dataset) * 0.8)
    num_val = int(len(eeg_dataset) * 0.1)
    num_test = len(eeg_dataset) - num_val - num_training

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(eeg_dataset, [num_training, num_val, num_test], generator=generator)
    
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
        test_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=opts.n_threads,
        pin_memory=True
    )
    
    prob_predictions, final_predictions, targets = stacking_classifier.test(test_loader_av, test_loader_eeg)
    
    loss = log_loss(targets, prob_predictions)
    
    print("test loss: ", loss)
    
    output_file = "results/result_test.csv"
    
    compute_confusion_matrix(targets, final_predictions, "./Images/confusion_matrix.pdf", "Confusion Matrix - Meta Model")
    accuracy = accuracy_score(targets, final_predictions, normalize=True)
    
    print("test accuracy: ", accuracy)
    
    data = [{"test": "Meta-classifier test", "len_dataset": len(av_dataset), "accuracy": f"{accuracy:.2f}", "loss":f"{loss:.2f}"}]
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["test", "len_dataset", "accuracy", "loss"])
    
        # Scrivi l'intestazione (header)
        writer.writeheader()
    
        # Scrivi i dati
        writer.writerows(data)
    
    
    
    