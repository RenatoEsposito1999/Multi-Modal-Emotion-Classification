import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.io import loadmat
import os

# The labels of the SEED_IV datasets
label_1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
label_2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
label_3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

# The channels of Epoc plus
channels_epoc_plus = [3, 4, 5, 7, 11, 13, 15, 21, 23, 31, 41, 49, 58, 60]

def pad_and_mask(data):
    """
    Pad the data to the length of the longest sequence and create a mask for non-padded elements.
    
    Args:
        data (list of np.ndarray): List of EEG sequences of varying lengths.
        
    Returns:
        padded_data (torch.Tensor): Tensor containing padded sequences.
        mask (torch.Tensor): Mask indicating valid elements (1 for real data, 0 for padding).
    """
    # Find the length of the longest sequence
    max_length = max(sequence.shape[0] for sequence in data)
    
    # Pad sequences with zeros
    padded_data = []
    mask = []
    for sequence in data:
        length = sequence.shape[0]
        padding = max_length - length
        padded_sequence = np.pad(sequence, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        padded_data.append(padded_sequence)
        mask.append([1] * length + [0] * padding)  # 1 for real data, 0 for padding
    return padded_data, mask

def preprocess(path):
    found_files = [file for file in ["EEGTrain.npz", "EEGVal.npz", "EEGTest.npz"] if os.path.exists(os.path.join("./EEG_data", file))]
    if len(found_files) != 3:
        data = []
        labels = []
        for i, folder in enumerate(os.listdir(path)):
            print(folder)
            if folder == "1":
                label = label_1
            elif folder == "2":
                label = label_2
            elif folder == "3":
                label = label_3
            for root, _, files in os.walk(os.path.join(path, folder)):
                for file in files:
                    if file.endswith(".mat"):
                        datamat = loadmat(os.path.join(path, folder, file))
                        index = 0
                        for key in datamat:
                            if not key.startswith('__'):
                                tmp = datamat[key]
                                data.append(tmp.T)                             
                                labels.append(label[index])
                                index += 1

        # Apply padding and masking
        padded_data, mask = pad_and_mask(data)
        
        padded_data=np.stack(padded_data)
        mask = np.stack(mask)
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        print("Padded Data Shape:", padded_data.shape)
        print("Mask Shape:", mask.shape)
        print("Labels Shape:", labels.shape)
    
        # Create dataset
        EEGDataset_complete = TensorDataset(padded_data, labels, mask)
        EEGDataset_train, EEGDataset_val, EEGDataset_test = torch.utils.data.random_split(EEGDataset_complete, [756, 216, 108])
        save_dataset_to_npz(EEGDataset_train, "./EEG_data/EEGTrain.npz")
        save_dataset_to_npz(EEGDataset_val, "./EEG_data/EEGVal.npz")
        save_dataset_to_npz(EEGDataset_test, "./EEG_data/EEGTest.npz")

def save_dataset_to_npz(subset, save_path):
    """
    Save the entire PyTorch Subset, containing both features and labels, into a single NumPy file.
 
    Args:
        subset (torch.utils.data.Subset): The dataset subset to save.
        save_path (str): File path to save as a single .npz file.
 
    Returns:
        None
    """
    features_list = []
    labels_list = []
    masks_list = []
 
    for features, label, mask in subset:
        features_list.append(features.numpy())  # Convert features to NumPy
        labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else label)  # Convert labels to NumPy
        masks_list.append(mask.numpy())  # Convert masks to NumPy
 
    # Convert to full NumPy arrays
    features_array = np.stack(features_list)  # Stack to create a 3D array
    labels_array = np.array(labels_list)  # 1D array for labels
    masks_array = np.stack(masks_list)  # Stack to create a 2D array for masks
 
    # Save all arrays into a single .npz file
    np.savez(save_path, features=features_array, labels=labels_array, masks=masks_array)
    print(f"Subset saved to {save_path}")
