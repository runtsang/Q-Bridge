import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

class _TensorUtils:
    """Utility to cast data to float32 tensors on device."""
    @staticmethod
    def as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        return torch.as_tensor(data, dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for a two‑branch autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_layers: int = 2  # depth of the variational circuit
    quantum_dropout: float = 0.05

class ClassicalAutoencoder(nn.Module):
    """Classical encoder–decoder head."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

class QuantumFeatureExtractor(nn.Module):
    """A lightweight surrogate for a variational quantum circuit."""
    def __init__(self, latent_dim: int, depth: int = 2) -> None:
        super().__init__()
        # Simple linear chain to emulate quantum parameterisation
        layers = []
        layers.append(nn.Linear(latent_dim, latent_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        self.core = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.core(z)

class QuantumAutoencoderHybrid(nn.Module):
    """Hybrid classical–quantum autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.classical = ClassicalAutoencoder(cfg)
        self.quantum = QuantumFeatureExtractor(cfg.latent_dim, depth=cfg.quantum_layers)
        # Decoder uses the same architecture as the classical decoder
        self.decoder = self.classical.decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.classical.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_q = self.quantum(z)
        return self.decoder(z_q)

def train_hybrid(
    model: QuantumAutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop that optimises both classical and quantum parts."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_TensorUtils.as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "ClassicalAutoencoder",
    "QuantumFeatureExtractor",
    "QuantumAutoencoderHybrid",
    "train_hybrid",
]
