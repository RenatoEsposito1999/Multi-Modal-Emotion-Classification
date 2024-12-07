import torch
import torch.nn as nn


class EEGCNNPreprocessor(nn.Module):
    def __init__(self, num_channels=14, d_model=128, cnn_out_channels=32):
        super(EEGCNNPreprocessor, self).__init__()
        self.d_model = d_model

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),  # Normalize CNN outputs
            nn.MaxPool1d(2,1),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),  # Normalize CNN outputs
            nn.MaxPool1d(2,1),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,1)
        )
        #(Idea): Removed the stride to include more temporal informations, included the maxpooling to reduce dimensionality in some way

    def forward(self, x):
        """
        Forward pass through the CNN.
        Args:
            x: Tensor of shape (batch_size, sequence_length, num_channels)
            mask: Tensor of shape (batch_size, sequence_length) indicating padding.
        """
        
        """
        Normalize the EEG input signal (z-score normalization).
        Args:
            x: Tensor of shape (batch_size, sequence_length, num_channels)
        """
        # Normalize across channels for each sample
        #mean = x.mean(dim=1, keepdim=True)  # Mean across sequence length (time)
        #std = x.std(dim=1, keepdim=True)  # Std across sequence length (time)
        #x = (x - mean) / (std + 1e-6)  # Avoid division by zero

        #(Idea): perform it on sample based not on batch

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_channels, sequence_length)
       

        # Pass through CNN
        x = self.cnn(x)  # Output shape: (batch_size, d_model, reduced_sequence_length)
        
        # Transpose back to (batch_size, sequence_length, d_model)
        x = x.permute(0, 2, 1)
       
        return x