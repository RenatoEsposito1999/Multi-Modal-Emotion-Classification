import numpy as np
import torch
from datasets.eeg_dataset import EEGDataset

def get_training_set_EEG():
    data = np.load("./EEG_data/EEGTrain.npz")
    features = data["features"]
    labels = data["labels"]
    mask=data["masks"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    return EEGDataset(features_tensor, labels_tensor, mask)


def get_validation_set_EEG():
    data = np.load("./EEG_data/EEGValidation.npz")
    features = data["features"]
    labels = data["labels"]
    mask=data["masks"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    return EEGDataset(features_tensor, labels_tensor, mask)


def get_test_set_EEG():
    data = np.load("./EEG_data/EEGTest.npz")
    features = data["features"]
    labels = data["labels"]
    mask=data["masks"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    return EEGDataset(features_tensor, labels_tensor, mask)