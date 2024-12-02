from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=128, num_channels=62, num_classes=4, data=None, labels=None):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.data = data
        self.labels = labels

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]