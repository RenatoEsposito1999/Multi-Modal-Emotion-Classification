import torch
import torch.nn as nn


class EEGCNNPreprocessor(nn.Module):
    def __init__(self, num_channels=62, d_model=128, num_heads=8, num_layers=4, sequence_length=256, cnn_out_channels=32):
        super(EEGCNNPreprocessor, self).__init__()
        
        self.input_channels = num_channels
        self.d_model = d_model


        # 1D CNN layer to process each sensor independently across time
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        

    def forward(self,x):
        x = x.permute(0, 2, 1)
        # Pass through CNN
        x = self.cnn(x)  # Shape: (batch_size, d_model, sequence_length)

        # Transpose back to (batch_size, sequence_length, d_model) for the transformer
        x = x.permute(0, 2, 1)

        return x