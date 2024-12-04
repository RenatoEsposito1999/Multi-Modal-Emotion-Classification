import torch
from torch.utils.data import Dataset
import random
import numpy as np

class Synchronized_data(Dataset):
    def __init__(self, dataloader):
        """
        Initialize the dataset by storing combined data and masks for each label.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader.
        """
        # Dictionary to store (data, mask) tuples for each label
        self.label_data = {0: [], 1: [], 2: [], 3: []}
        
        # Populate the label_data dictionary
        for batch_data, labels in dataloader:
            for i in range(len(labels)):
                label = labels[i].item()
                self.label_data[label].append(batch_data[i])
                
    def pad_and_mask(self, sequence, max_length):
        """
        Pad a single sequence to the given max_length and create a mask.
        """
        length = sequence.shape[0]
        padding = max_length - length
        padded_sequence = np.pad(sequence, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        mask = [1] * length + [0] * padding  # 1 for real data, 0 for padding
        return padded_sequence, mask


    # Define custom collate function for padding and masking
    def collate_fn(self, datas):
        max_sequence_length=0
        for data in datas:
            max_sequence_length = max(data.shape[0], max_sequence_length)
        padded_data, masks = zip(*[self.pad_and_mask(data, max_sequence_length) for data in datas])
            
        # Convert to tensors
        padded_data = torch.tensor(padded_data, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        
            
        return padded_data, masks
    
    def generate_artificial_batch(self, labels):
        """
        Generate an artificial batch based on specified labels.
        A single sample is selected randomly for each label.
        
        Args:
            labels (list): List of label indices to sample from.
        
        Returns:
            Tensor: A tensor of shape [batch_size, 2, sequence_len, features], where:
                    - dim 1 = 0 contains the data.
                    - dim 1 = 1 contains the masks.
        """
        artificial_data = []
        
        for label in labels:
            print(type(label.item()))
            print(label.item())
            if not self.label_data[label.item()]:
                raise ValueError(f"No data available for label {label}")
            
            # Randomly select one data point for this label
            random_data = random.choice(self.label_data[label.item()])
            artificial_data.append(random_data)
        
        data, mask = self.collate_fn(artificial_data)
        
        
        # Convert the data and masks to tensors
        data_tensor = torch.stack(data)  # Shape: [batch_size, sequence_len, features]
        mask_tensor = torch.stack(mask)  # Shape: [batch_size, sequence_len, features]
        
        print("Shape data: ", data_tensor.shape)
        print("Shape mask: ", mask_tensor)
        
        # Combine data and mask into a single tensor with an extra dimension
        #combined_tensor = torch.stack([data_tensor, mask_tensor], dim=1)  # Shape: [batch_size, 2, sequence_len, features]
        

        ### Abbasso l'annotation.txt

        return data_tensor,mask_tensor
    


