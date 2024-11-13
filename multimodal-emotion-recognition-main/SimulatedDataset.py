import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimulatedEEGDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=128, num_channels=64, num_classes=4, noise_level=0.1):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.noise_level = noise_level
        
        # Generate random EEG-like data with sine waves and noise
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        
        
        for i in range(self.num_samples):
            # Simulate a class label for each sample
            label = np.random.randint(0, self.num_classes)
            
            # Generate base frequency component based on class label
            freq = (label + 1) * 0.1  # Frequency differs per class
            
            # Generate a time series for each channel
            sample = np.zeros((self.sequence_length, self.num_channels))
            time = np.linspace(0, 2 * np.pi, self.sequence_length)

            for ch in range(self.num_channels):
                # Create a unique sine wave for each channel with added noise
                signal = np.sin(freq * time + ch * 0.1)  # Slight phase shift per channel
                noise = self.noise_level * np.random.randn(self.sequence_length)  # Add noise
                sample[:, ch] = signal + noise

            data.append(sample)
            labels.append(label)

        data = np.stack(data)  # Shape: (num_samples, sequence_length, num_channels)
        labels = np.array(labels)  # Shape: (num_samples,)
        
        print("Data: ", data.shape)
        print("Lables: ", labels.shape)
        
        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        
        return data, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

x = SimulatedEEGDataset()
