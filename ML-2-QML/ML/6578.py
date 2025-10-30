import torch
from torch import nn

class QCNNEnhanced(nn.Module):
    """
    Residual QCNN‑inspired network with deeper fully‑connected blocks.
    The architecture mirrors the original seed but adds a skip connection around
    the middle layers and an extra projection before the sigmoid head, improving
    gradient flow for deeper models.
    """
    def __init__(self,
                 in_features: int = 8,
                 hidden_sizes: tuple[int,...] = (16, 16, 12, 8),
                 final_dim: int = 4) -> None:
        """
        Parameters
        ----------
        in_features: int
            Dimensionality of the input vector.
        hidden_sizes: tuple[int,...]
            Sizes of the hidden layers for the convolutional part.
        final_dim: int
            Size of the representation fed to the head.
        """
        super().__init__()

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_sizes[0]),
            nn.Tanh()
        )

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh()
        )

        # Residual connection: conv1 -> conv2
        self.residual = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        # Pooling layers
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_sizes[3], final_dim),
            nn.Tanh()
        )

        # Classification head
        self.head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Supports batched inputs.
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        conv_out = self.conv2(x)
        # Add residual skip
        x = conv_out + self.residual(conv_out)
        x = self.pool1(x)
        x = self.pool2(x)
        x = torch.sigmoid(self.head(x))
        return x
