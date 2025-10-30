import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvGen126(nn.Module):
    """
    Hybrid convolutional module that can operate in classical mode.
    It combines a convolution, a fully‑connected transform, an LSTM,
    and a classifier head, mirroring the structure of the anchor Conv.py
    but with additional layers.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_features: int = 10,
        hidden_dim: int = 20,
        vocab_size: int = 100,
        tagset_size: int = 5,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolution layer
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Fully‑connected layer that takes the mean of the conv output
        self.fc = nn.Linear(1, 1)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Classifier head
        layers = []
        in_dim = hidden_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classical mode.

        Args:
            x: Tensor of shape (batch, 1, H, W) where H=W=kernel_size.

        Returns:
            logits: Tensor of shape (batch, 2)
        """
        # Convolution
        conv_out = self.conv(x)
        # Thresholded sigmoid activation
        conv_act = torch.sigmoid(conv_out - self.threshold)
        # Reduce to a scalar per sample
        mean = conv_act.mean(dim=[2, 3], keepdim=True)
        # Fully‑connected transform
        fc_out = self.fc(mean).squeeze(-1)
        # LSTM expects (batch, seq_len, features)
        lstm_in = fc_out.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        # Classifier
        logits = self.classifier(lstm_out.squeeze(1))
        return logits
