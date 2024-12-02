import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os


 
# The labels of the SEED_IV datasets
label_1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
label_2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
label_3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

channels_epoc_plus = [3,4,5,7,11,13,15,21,23,31,41,49,58,60]
 
class EEGDataset(Dataset):
    """
    Lazy-loading EEG dataset for each .mat file with 24 EEG sequences and corresponding labels.
    """
    def __init__(self, path):
        self.data= []
        self.labels = []
        # Iterate through each folder in the path
        for i,folder in enumerate(os.listdir(path)):
            if folder == "1":
                label_list = label_1
            elif folder == "2":
                label_list = label_2
            elif folder == "3":
                label_list = label_3
            else:
                continue
            # Iterate through all files in the folder
            for root, _, files in os.walk(os.path.join(path, folder)):
                for file in files:
                    if file.endswith(".mat"):
                        datamat = loadmat(path + "/" + folder + "/" + file)
                        index = 0                     
                        for key in datamat:
                            if not key.startswith('__'):
                                tmp = datamat[key]
                                for i in range(tmp.shape[0]):
                                    if i in channels_epoc_plus:
                                        # Inizializza l'array di output
                                        self.data.append(tmp[i].T)                             
                                self.labels.append(label_list[index])
                                index += 1
                                
    def __init__(self, data, labels, mask):
        self.data = data
        self.labels = labels
        self.mask = mask
                        
                                
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        mask = self.mask[idx]
        
        return data, label, mask