import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    circuit_depth: int = 3
    num_trash: int = 2

class HybridAutoencoder(nn.Module):
    def __init__(self, cfg: HybridAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        # Classical encoder
        encoder_layers = []
        dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(dim, h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(cfg.dropout))
            dim = h
        encoder_layers.append(nn.Linear(dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Quantum decoder simulation parameters
        self.circuit_depth = cfg.circuit_depth
        self.num_trash = cfg.num_trash
        # Classical decoder
        decoder_layers = []
        dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(cfg.dropout))
            dim = h
        decoder_layers.append(nn.Linear(dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_decode(self, z: torch.Tensor) -> torch.Tensor:
        import numpy as np
        batch_size, latent_dim = z.shape
        angles_np = np.random.default_rng(42).random((self.circuit_depth, latent_dim))
        angles = torch.tensor(angles_np, dtype=z.dtype, device=z.device)
        out = z.clone()
        for i in range(self.circuit_depth):
            out = torch.cos(angles[i]) * out + torch.sin(angles[i]) * torch.flip(out, dims=[1])
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        quantum_out = self.quantum_decode(z)
        return self.decoder(quantum_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def HybridAutoencoderFactory(input_dim: int, *, latent_dim: int = 32,
                             hidden_dims: Tuple[int, int] = (128, 64),
                             dropout: float = 0.1,
                             circuit_depth: int = 3,
                             num_trash: int = 2) -> HybridAutoencoder:
    cfg = HybridAutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout, circuit_depth, num_trash)
    return HybridAutoencoder(cfg)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_hybrid_autoencoder(model: HybridAutoencoder, data: torch.Tensor,
                            *, epochs: int = 100, batch_size: int = 64,
                            lr: float = 1e-3, device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "train_hybrid_autoencoder", "HybridAutoencoderConfig"]
