import numpy as np
import torch
from datasets.eeg_dataset import EEGDataset

def get_training_set_EEG():
    data = np.load("./EEG_data/EEGTrain.npz")
    features = data["features"]
    labels = data["labels"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    num_samples = features.shape[0]
    sequence_length = features.shape[1]
    num_channels = features.shape[2]
    return EEGDataset(num_samples=num_samples, sequence_length=sequence_length, num_channels=num_channels,data=features_tensor, labels=labels_tensor)


def get_validation_set_EEG():
    data = np.load("./EEG_data/EEGVal.npz")
    features = data["features"]
    labels = data["labels"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    num_samples = features.shape[0]
    sequence_length = features.shape[1]
    num_channels = features.shape[2]
    return EEGDataset(num_samples=num_samples, sequence_length=sequence_length, num_channels=num_channels,data=features_tensor, labels=labels_tensor)


def get_test_set_EEG():
    data = np.load("./EEG_data/EEGTest.npz")
    features = data["features"]
    labels = data["labels"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    num_samples = features.shape[0]
    sequence_length = features.shape[1]
    num_channels = features.shape[2]
    return EEGDataset(num_samples=num_samples, sequence_length=sequence_length, num_channels=num_channels,data=features_tensor, labels=labels_tensor)