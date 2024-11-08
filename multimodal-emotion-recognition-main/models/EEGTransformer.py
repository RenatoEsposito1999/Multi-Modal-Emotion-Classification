import torch
import torch.nn as nn
import math

class EEGCNNTransformerEncoder(nn.Module):
    def __init__(self, num_channels=64, d_model=128, num_heads=8, num_layers=4, sequence_length=256, cnn_out_channels=32):
        super(EEGCNNTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # 1D CNN layer to process each sensor independently across time
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Positional encoding for the transformer input
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(sequence_length, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Forward pass for the EEG CNN + Transformer Encoder model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, num_channels)
        
        Returns:
            Tensor of shape (batch_size, d_model) - the final embedding
        """
        # Transpose to (batch_size, num_channels, sequence_length) for CNN processing
        x = x.permute(0, 2, 1)
        
        # Pass through CNN
        x = self.cnn(x)  # Shape: (batch_size, d_model, sequence_length)

        # Transpose back to (batch_size, sequence_length, d_model) for the transformer
        x = x.permute(0, 2, 1)

        # Add positional encoding
        x = x + self.positional_encoding[:self.sequence_length, :]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Return the final embedding
        return x.mean(dim=1)  # Aggregates across time steps to get a fixed-size embedding

    def _generate_positional_encoding(self, length, d_model):
        """
        Generates a positional encoding matrix of shape (length, d_model).
        """
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add a batch dimension