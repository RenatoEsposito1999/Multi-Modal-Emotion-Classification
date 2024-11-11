import torch
import torch.nn as nn

def conv1d_block_eeg(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2, 1)
    )

class EEGCNNPreprocessor(nn.Module):
    def __init__(self, input_channels=64, d_model=128):
        super(EEGCNNPreprocessor, self).__init__()
        
        self.input_channels = input_channels
        self.d_model = d_model

        # Define CNN layers
        self.conv1d_0 = conv1d_block_eeg(self.input_channels, 32)
        self.conv1d_1 = conv1d_block_eeg(32, 64)
        self.conv1d_2 = conv1d_block_eeg(64, self.d_model)

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        return x
