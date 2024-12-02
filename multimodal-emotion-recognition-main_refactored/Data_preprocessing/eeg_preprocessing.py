import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.io import loadmat
from scipy import interpolate
import os

#The labels of the SEED_IV datasets
label_1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
label_2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
label_3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

#The channels of Epoc plus
channels_epoc_plus = [3,4,5,7,11,13,15,21,23,31,41,49,58,60]

'''
    In this function checks if the three files are present, otherwise creates them.
    In particular apply an interpolation process in order to reduce the sequence length of the data, and adapt
    them for the model.
    
    Args:
        - Path = the path of eeg data
        - sequence_length = the sequence length that you want, this must be equal to the sequence length trained by the model
    
    Returns:
        None
'''
def preprocess(path):
    found_files = [file for file in ["EEGTrain.npz", "EEGVal.npz", "EEGTest.npz"] if os.path.exists(os.path.join("./EEG_data", file))]
    if len(found_files) != 3:
        data = []
        labels = []
        for i,folder in enumerate(os.listdir(path)):
            if folder == "1":
                label = label_1
            elif folder == "2":
                label = label_2
            elif folder == "3":
                label = label_3
            for root, _ , files in os.walk(os.path.join(path, folder)):
                print("PATH: ", path)
                for file in files:
                    if file.endswith(".mat"):
                        datamat = loadmat(path + "/" + folder + "/" + file)
                        index = 0                     
                        for key in datamat:
                            if not key.startswith('__'):
                                tmp = datamat[key]
                                # Inizializza l'array di output
                                data.append(tmp.T)                             
                                labels.append(label[index])
                                index += 1
        
            
        data = np.stack(data)  # Shape: (num_samples, sequence_length, num_channels)
        labels = np.array(labels) # Shape: (num_samples,)
            
        print("Data: ", data.shape)
        print("Lables: ", labels.shape)
    
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        EEGDataset_complete = TensorDataset(data, labels)
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
    
    

        
        


    
    
    
    


        
        
         
        
    