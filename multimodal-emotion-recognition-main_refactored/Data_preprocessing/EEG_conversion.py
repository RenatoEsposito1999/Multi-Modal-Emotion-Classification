import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import os

# The labels of the SEED_IV datasets
label_1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
label_2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
label_3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

def pad_and_mask(sequence, max_length):
    """
    Pad a single sequence to the given max_length and create a mask.
    """
    length = sequence.shape[0]
    padding = max_length - length
    padded_sequence = np.pad(sequence, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    mask = [1] * length + [0] * padding  # 1 for real data, 0 for padding
    return padded_sequence, mask

class EEGDataset(Dataset):
    """
    Lazy-loading EEG dataset.
    """
    def __init__(self, path):
        self.data_paths = []
        self.labels = []
        for folder in os.listdir(path):
            if folder == "1":
                label_list = label_1
            elif folder == "2":
                label_list = label_2
            elif folder == "3":
                label_list = label_3
            else:
                continue
            for root, _, files in os.walk(os.path.join(path, folder)):
                for file in files:
                    if file.endswith(".mat"):
                        self.data_paths.append(os.path.join(root, file))
                        self.labels.append(label_list[len(self.data_paths) - 1])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load the .mat file and extract EEG data
        datamat = loadmat(data_path)
        for key in datamat:
            if not key.startswith('__'):
                sequence = datamat[key].T
                break

        return sequence, label

def preprocess(path, batch_size=32, max_sequence_length=500):
    """
    Preprocess EEG data with lazy loading and dynamic padding in batches.
    """
    dataset = EEGDataset(path)

    # Define custom collate function for padding and masking
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        padded_sequences, masks = zip(*[pad_and_mask(seq, max_sequence_length) for seq in sequences])
        
        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return padded_sequences, labels, masks

    # Create DataLoader for lazy loading
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader

if __name__ == "__main__":
    path_to_data = "./EEG_data"  # Replace with actual path
    batch_size = 32
    max_sequence_length = 500  # Set a maximum length for sequences (truncate or pad)
    dataloader = preprocess(path_to_data, batch_size=batch_size, max_sequence_length=max_sequence_length)
    
    for batch_idx, (data, labels, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Data Shape:", data.shape)  # Shape: (batch_size, max_sequence_length, num_channels)
        print("Labels Shape:", labels.shape)  # Shape: (batch_size,)
        print("Mask Shape:", masks.shape)  # Shape: (batch_size, max_sequence_length)
        break  # Process only the first batch for demonstration
