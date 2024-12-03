import torch
from torch.utils.data import Dataset
import random

class SynchronizedDataset(Dataset):
    def __init__(self, dataloader):
        """
        Initialize the dataset by storing combined data and masks for each label
        
        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader
        """
        # Dictionary to store (data, mask) tuples for each label
        self.label_data = {0: [], 1: [], 2: [], 3: []}
        
        # Populate the label_data dictionary
        for batch_data, labels, masks in dataloader:
            for i in range(len(labels)):
                label = labels[i].item()
                self.label_data[label].append((batch_data[i], masks[i]))
    
    def generate_artificial_batch(self, labels):
        """
        Generate an artificial batch based on specified labels
        
        Args:
            labels (list): List of label indices to sample from
        
        Returns:
            list: List of (data, mask) tuples
        """
        artificial_batch = []
        
        for label in labels:
            # Randomly select a data point for this label
            if not self.label_data[label]:
                raise ValueError(f"No data available for label {label}")
            
            random_data = random.choice(self.label_data[label])
            artificial_batch.append(random_data)
        
        return artificial_batch