import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os


# The labels of the SEED_IV datasets
label_map = {
    "1": [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    "2": [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    "3": [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def preprocess_eeg(path):
    """
    Initialize the dataset.
        
    Args:
        path (str): Path to the directory containing EEG data organized in subfolders.
        selected_channels (list): List of channel indices to include.
    """
    file_paths = []
    labels = []

    # Collect file paths and associated labels
    for folder in os.listdir(path):
        if folder in label_map:
            label_list = label_map[folder]
            folder_path = os.path.join(path, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".mat"):
                        datamat = loadmat(path + "/" + folder + "/" + file)
                        index = 0                     
                        for key in datamat:
                            if not key.startswith('__'):
                                tmp = datamat[key]      
                                np.save(f"./EEG_data/{folder}_{key}_{label_list[index]}", np.array(tmp))                             
                                index += 1


preprocess_eeg("C:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/datasets/SEED_IV/SEED_IV/eeg_raw_data")

