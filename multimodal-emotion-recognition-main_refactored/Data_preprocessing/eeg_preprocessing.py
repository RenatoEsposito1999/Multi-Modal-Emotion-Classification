import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.io import loadmat
from scipy import interpolate
from torch.utils.data import DataLoader
import os
from datasets.eeg_dataset import EEGDataset

#The labels of the SEED_IV datasets
label_1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
label_2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
label_3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

#The channels of Epoc plus
channels_epoc_plus = [3,4,5,7,11,13,15,21,23,31,41,49,58,60]

'''
    In this function checks if the three files are present continue, otherwise creates them.
   
    Args:
        - Path = the path of eeg data
    
    Returns:
        None
'''
def preprocess(path, opt):
    found_files = [file for file in ["EEGTrain.npz", "EEGValidation.npz", "EEGTest.npz"] if os.path.exists(os.path.join("./EEG_data", file))]
    if len(found_files) != 3:
        EEGDataset_complete = EEGDataset(path)
        max_sequence_length = 0
        for idx in range(len(EEGDataset_complete)):
            sequence, _ = EEGDataset_complete[idx]
            max_sequence_length = max(max_sequence_length, sequence.shape[0])
        print(f"Dynamic Max Sequence Length: {max_sequence_length}")
        
         # Define custom collate function for padding and masking
        def collate_fn(batch):
            sequences, labels = zip(*batch)
            padded_sequences, masks = zip(*[pad_and_mask(seq, max_sequence_length) for seq in sequences])
        
            # Convert to tensors
            padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
            return padded_sequences, labels, masks

        train_split, validation_split, test_split = torch.utils.data.random_split(EEGDataset_complete, [756, 216, 108])
        
        dataloader = DataLoader(EEGDataset_complete, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        
        train_loader = DataLoader(train_split, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(validation_split, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_split, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
        
        save_split_to_npz(train_loader, "./EEG_data/EEGTrain.npz")
        save_split_to_npz(val_loader,"./EEG_data/EEGValidation.npz")
        save_split_to_npz(test_loader,"./EEG_data/EEGTest.npz")
        
        


def save_split_to_npz(loader, save_path):
    """
    Save DataLoader content into an NPZ file.
    """
    all_features = []
    all_labels = []
    all_masks = []
    
    for data, labels, masks in loader:
        all_features.append(data.numpy())
        all_labels.append(labels.numpy())
        all_masks.append(masks.numpy())
    
    # Combine all batches into single arrays
    features_array = np.concatenate(all_features, axis=0)  # Shape: (total_samples, max_sequence_length, num_channels)
    labels_array = np.concatenate(all_labels, axis=0)  # Shape: (total_samples,)
    masks_array = np.concatenate(all_masks, axis=0)  # Shape: (total_samples, max_sequence_length)
    
    # Save to NPZ
    np.savez(save_path, features=features_array, labels=labels_array, masks=masks_array)
    print(f"Saved DataLoader to {save_path}")
    


def pad_and_mask(sequence, max_length):
    """
    Pad a single sequence to the given max_length and create a mask.
    """
    length = sequence.shape[0]
    padding = max_length - length
    padded_sequence = np.pad(sequence, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    mask = [1] * length + [0] * padding  # 1 for real data, 0 for padding
    return padded_sequence, mask

        
        


    
    
    
    


        
        
         
        
    