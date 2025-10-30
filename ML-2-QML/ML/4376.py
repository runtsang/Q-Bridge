"""Hybrid classical regression model combining autoencoding, convolutional filtering,
and RBF kernel features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _as_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# 1. Data generation – superposition + optional convolutional filtering
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int,
                                kernel_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from a simple sinusoidal function and optionally
    apply a 2‑D convolutional filter to each sample reshaped to ``kernel_size``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    # If the feature vector can be reshaped into a square, apply the Conv filter
    if num_features == kernel_size ** 2:
        conv = Conv()
        features = np.stack([conv.run(sample.reshape(kernel_size, kernel_size))
                             for sample in x], axis=0)
    else:
        features = x

    return features, y.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and a target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": _as_tensor(self.features[idx]),
            "target": _as_tensor(self.labels[idx]),
        }


# --------------------------------------------------------------------------- #
# 3. Classical building blocks
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int,...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class ConvFilter(nn.Module):
    """2‑D convolutional filter emulating a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class RBFKernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# --------------------------------------------------------------------------- #
# 4. Hybrid regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """
    Combines an autoencoder, a convolutional filter, and an RBF kernel
    to produce a regression output.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 kernel_dim: int = 10,
                 gamma: float = 1.0):
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim=latent_dim)
        self.conv = ConvFilter()
        self.kernel = RBFKernel(gamma=gamma)
        # Basis vectors used in kernel feature mapping
        self.basis = nn.Parameter(torch.randn(kernel_dim, latent_dim))
        self.head = nn.Linear(kernel_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode to latent space
        z = self.autoencoder.encode(x)

        # Convolutional filtering (optional – can be omitted if not needed)
        if z.shape[-1] >= 4:
            conv_vec = torch.tensor([self.conv.run(z[i].view(2, 2).cpu().numpy())
                                     for i in range(z.shape[0])],
                                    dtype=torch.float32,
                                    device=z.device)
            conv_vec = conv_vec.unsqueeze(-1)  # shape (batch, 1)
            z = torch.cat([z, conv_vec], dim=-1)

        # Kernel similarity with learned basis
        k_vecs = []
        for i in range(z.shape[0]):
            k_row = torch.stack([self.kernel(z[i:i+1], b.unsqueeze(0))
                                 for b in self.basis], dim=0)
            k_vecs.append(k_row.squeeze(-1))
        k_mat = torch.stack(k_vecs, dim=0)  # shape (batch, kernel_dim)

        out = self.head(k_mat)
        return out.squeeze(-1)


def train_hybrid_regression(model: HybridRegressionModel,
                            dataset: Dataset,
                            epochs: int = 100,
                            batch_size: int = 64,
                            lr: float = 1e-3,
                            device: torch.device | None = None) -> list[float]:
    """Simple training loop that returns the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(states)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "AutoencoderNet",
    "ConvFilter",
    "RBFKernel",
    "HybridRegressionModel",
    "train_hybrid_regression",
]
