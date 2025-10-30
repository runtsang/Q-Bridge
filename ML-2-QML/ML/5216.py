import torch
from torch import nn
import numpy as np
from typing import Iterable, Tuple

class QuantumKernel(nn.Module):
    """Classical RBF kernel mimicking a quantum kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 convolutional filter inspired by the quanvolution example."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class FullyConnectedQuantumLayer(nn.Module):
    """Classical stand‑in for a fully‑connected quantum layer."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(theta)).mean(dim=0, keepdim=True)

class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder combining MLP, quanvolution, and quantum kernel."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 use_quanvolution: bool = False,
                 use_quantum_kernel: bool = False,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_quantum_kernel = use_quantum_kernel

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Optional quanvolution for image data
        self.qfilter = QuanvolutionFilter() if use_quanvolution else None

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Quantum kernel for regularisation
        self.kernel = QuantumKernel(gamma=kernel_gamma) if use_quantum_kernel else None

        # Quantum fully‑connected layer placeholder
        self.qfc = FullyConnectedQuantumLayer()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.qfilter:
            z = self.qfilter(z.unsqueeze(1)).squeeze(1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        out = out + self.qfc(out[:, -1:].detach())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_hybrid_autoencoder(model: HybridAutoencoder,
                             data: torch.Tensor,
                             *,
                             epochs: int = 100,
                             batch_size: int = 64,
                             lr: float = 1e-3,
                             weight_decay: float = 0.0,
                             device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = mse(recon, batch)

            if model.kernel:
                z = model.encode(batch)
                K = model.kernel(z, z)
                loss += 0.01 * K.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
