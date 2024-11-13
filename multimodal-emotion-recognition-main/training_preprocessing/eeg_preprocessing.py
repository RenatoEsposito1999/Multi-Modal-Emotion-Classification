import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy import interpolate
import os
#"C:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/datasets/SEED_IV/SEED_IV/eeg_raw_data/1/1_20160518.mat"

label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]

class EEGDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=128, num_channels=64, num_classes=4, noise_level=0.1, path=None):
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
        for root, _ , files in os.walk(self.path):
            for file in files:
                if file.endswith(".mat"):
                    datamat = loadmat(self.path + file)
                    index = 0
                    for key in datamat:
                        if not key.startswith('__'):
                            tmp = datamat[key]
                            # Inizializza l'array di output
                            downsampled_data = np.zeros((tmp.shape[0], self.sequence_length))
                            for i in range(tmp.shape[0]):
                                # Genera gli indici originali e quelli target
                                original_indices = np.linspace(0, 1, tmp.shape[1])
                                target_indices = np.linspace(0, 1, self.sequence_length)
                                # Crea la funzione di interpolazione e applicala agli indici target
                                interp_func = interpolate.interp1d(original_indices, tmp[i], kind='linear')
                                downsampled_data[i] = interp_func(target_indices)
                            
                            labels.append(label[index])
                            data.append(downsampled_data.T)
                            index += 1
                
        print(downsampled_data.T.shape)  # (62, 128)
        
        data = np.stack(data)  # Shape: (num_samples, sequence_length, num_channels)
        labels = np.array(labels)  # Shape: (num_samples,)
        
        print("Data: ", data.shape)
        print("Lables: ", labels.shape)
 
        
        return data, labels
    


        
        
         
        
    