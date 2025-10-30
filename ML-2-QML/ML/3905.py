import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCL(nn.Module):
    """
    Hybrid classical‑quantum fully connected layer.

    The forward pass first extracts spatial features with a shallow CNN,
    then projects to a vector of length ``n_features`` that will
    serve as parameters for the quantum circuit.  The class is fully
    trainable and can be used as a drop‑in replacement for the
    original FCL module.
    """

    def __init__(self,
                 in_channels: int = 1,
                 n_features: int = 4,
                 conv_channels: int = 8,
                 kernel_size: int = 3,
                 pool_kernel: int = 2,
                 fc_hidden: int = 64,
                 dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel),
            nn.Conv2d(conv_channels, 2 * conv_channels, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel),
        )
        # Compute flattened size after conv+pool
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        out = self.feature_extractor(dummy_input)
        flattened_size = out.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden, n_features)
        )
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a tensor of shape (batch, n_features)
        that can be interpreted as parameters for the quantum circuit.
        """
        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        logits = self.fc(flattened)
        return self.batch_norm(logits)

    def generate_params(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns the parameters for a batch of
        inputs.  Equivalent to calling ``self.forward``.
        """
        return self.forward(x)
