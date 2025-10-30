import torch
from torch import nn

class QCNN(nn.Module):
    """
    Classical convolution-inspired neural network with dropout and batch normalization.
    Mirrors the original QCNN architecture but adds regularization for improved
    generalisation.
    """

    def __init__(self,
                 dropout: float = 0.2,
                 use_batchnorm: bool = True) -> None:
        super().__init__()
        layers = []

        # Feature map
        layers += [nn.Linear(8, 16)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # First convolutional block
        layers += [nn.Linear(16, 16)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # First pooling block
        layers += [nn.Linear(16, 12)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(12))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Second convolutional block
        layers += [nn.Linear(12, 8)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(8))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Second pooling block
        layers += [nn.Linear(8, 4)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(4))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Third convolutional block
        layers += [nn.Linear(4, 4)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(4))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Output head
        layers.append(nn.Linear(4, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a sigmoid‑activated probability."""
        return torch.sigmoid(self.net(x))

def create_qcnn() -> QCNN:
    """Factory returning a default‑configured QCNN instance."""
    return QCNN()

__all__ = ["QCNN", "create_qcnn"]
