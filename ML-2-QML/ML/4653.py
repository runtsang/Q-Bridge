import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Classical radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder used as a reconstruction regulariser."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class HybridAutoEncoderKernelQCNet(nn.Module):
    """Classical hybrid model combining CNN, kernel, autoencoder and a sigmoid head."""
    def __init__(self,
                 in_channels: int = 3,
                 num_qubits: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Flatten size depends on input resolution; 32x32 RGB gives 55815
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical hybrid head
        self.hybrid_head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        # Autoencoder for reconstruction regularisation
        self.autoencoder = AutoencoderNet(1, latent_dim, hidden_dims)

        # Classical kernel module
        self.kernel = RBFKernel(kernel_gamma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        probs = self.hybrid_head(x)
        probs = torch.cat([probs, 1 - probs], dim=-1)

        # Reconstruction
        recon = self.autoencoder(x)

        return probs, recon

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two sets of feature vectors using RBF kernel."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridAutoEncoderKernelQCNet"]
