import torch
from torch.utils.data import Dataset
import random

class SynchronizedDataset(Dataset):
    def __init__(self, dataloader):
        """
        Initialize the dataset by storing combined data and masks for each label.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader.
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
        Generate an artificial batch based on specified labels.
        A single sample is selected randomly for each label.
        
        Args:
            labels (list): List of label indices to sample from.
        
        Returns:
            Tensor: A tensor of shape [2, len(labels), max(data_dim,max_dim)], where:
                    - dim 0 = 0 contains the data.
                    - dim 0 = 1 contains the masks.
        """
        artificial_data = []
        artificial_masks = []
        
        for label in labels:
            if not self.label_data[label]:
                raise ValueError(f"No data available for label {label}")
            
            # Randomly select one data point for this label
            random_data = random.choice(self.label_data[label])
            data, mask = random_data  # Unpack into data and mask
            
            artificial_data.append(data)
            artificial_masks.append(mask)
        
        # Convert the data and masks to tensors
        data_tensor = torch.stack(artificial_data)  # Shape: [len(labels), ...]
        mask_tensor = torch.stack(artificial_masks)  # Shape: [len(labels), ...]

        # Combine data and mask into a single tensor with an extra dimension
        combined_tensor = torch.stack([data_tensor, mask_tensor], dim=0)  # Shape: [2, len(labels), ...]

        return combined_tensor
