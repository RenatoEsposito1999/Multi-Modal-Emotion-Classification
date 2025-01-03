import torch
import random
from torcheeg.datasets import SEEDIVDataset
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LIST, SEED_IV_CHANNEL_LOCATION_DICT
from torcheeg import transforms

epoc_plus_channels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'FC5', 'FC6']
Filtered_channels = transforms.PickElectrode.to_index_list(epoc_plus_channels, SEED_IV_CHANNEL_LIST)
filtered_location_dict = {ch: SEED_IV_CHANNEL_LOCATION_DICT[ch] for ch in epoc_plus_channels}


def generate_dataset_SEEDIV(path, cached_save):
    
    dataset = SEEDIVDataset(
            io_path = cached_save if cached_save else None,
            root_path=path,
            offline_transform=transforms.Compose([
                transforms.PickElectrode(Filtered_channels),
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(filtered_location_dict)]),
            online_transform=transforms.Compose([
                transforms.ToTensor()]),
            label_transform=transforms.Compose([
                transforms.Select('emotion')
            ]),
            num_worker=16,
            io_size=167772160
        )
    
    return dataset



