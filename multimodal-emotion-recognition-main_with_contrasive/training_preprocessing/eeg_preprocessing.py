import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from scipy.io import loadmat
from scipy import interpolate
import os
#path = "C:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/datasets/SEED_IV/SEED_IV/eeg_raw_data/1/"

label_1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
label_2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
label_3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
channels_epoc_plus = [3,4,5,7,11,13,15,21,23,31,41,49,58,60]

class EEGDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=128, num_channels=62, num_classes=4, noise_level=0.1, path=None):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.noise_level = noise_level
        self.path = path
        
        self.data, self.labels = self.preprocess()
        
        
        
    def preprocess(self):
        data = []
        labels = []
        for i,folder in enumerate(os.listdir(self.path)):
            if folder == "1":
                label = label_1
            elif folder == "2":
                label = label_2
            elif folder == "3":
                label = label_3
            for root, _ , files in os.walk(os.path.join(self.path, folder)):
                print("PATH: ", self.path)
                for file in files:
                    if file.endswith(".mat"):
                        datamat = loadmat(self.path + "/" + folder + "/" + file)
                        index = 0
                        
                        for key in datamat:
                            if not key.startswith('__'):
                                tmp = datamat[key]
                                # Inizializza l'array di output
                                downsampled_data = np.zeros((14, self.sequence_length))
                                idx = -1
                                for i in range(tmp.shape[0]):
                                    if i in channels_epoc_plus:
                                        idx = idx + 1
                                        # Genera gli indici originali e quelli target
                                        original_indices = np.linspace(0, 1, tmp.shape[1])
                                        target_indices = np.linspace(0, 1, self.sequence_length)
                                        # Crea la funzione di interpolazione e applicala agli indici target
                                        interp_func = interpolate.interp1d(original_indices, tmp[i], kind='linear')
                                        downsampled_data[idx] = interp_func(target_indices)
                                
                                labels.append(label[index])
                                data.append(downsampled_data.T)
                                index += 1
                
        print(downsampled_data.T.shape)  # (128,14)
        
        data = np.stack(data)  # Shape: (num_samples, sequence_length, num_channels)
        labels = np.array(labels) # Shape: (num_samples,)
        
        print("Data: ", data.shape)
        print("Lables: ", labels.shape)
 
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def create_dataset_from_file_npz(file):
    data = np.load(file)
    features = data["features"]
    labels = data["labels"]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(features_tensor, labels_tensor)

def save_dataset_to_npz(subset, save_path):
    """
    Save the entire PyTorch Subset, containing both features and labels, into a single NumPy file.
 
    Args:
        subset (torch.utils.data.Subset): The dataset subset to save.
        save_path (str): File path to save as a single .npz file.
 
    Returns:
        None
    """
    # Extract all data from the subset
    features_list = []
    labels_list = []
 
    for features, label in subset:
        features_list.append(features.numpy())  # Convert features to NumPy
        labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else label)  # Convert labels to NumPy
 
    # Convert to full NumPy arrays
    features_array = np.stack(features_list)  # Stack to create a 2D array
    labels_array = np.array(labels_list)  # 1D array for labels
 
    # Save both features and labels into a single .npz file
    np.savez(save_path, features=features_array, labels=labels_array)
 
    print(f"Subset saved to {save_path}")
    
    

        
        


    
    
    
    


        
        
         
        
    