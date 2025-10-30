import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Callable, List, Optional

import numpy as np

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    residual: bool = False
    random_sampler: bool = True
    sampler_std: float = 1.0
    device: torch.device | None = None

class ResidualDenseBlock(nn.Module):
    """Simple residual dense block with two linear layers."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(self.relu(self.linear1(x)))

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with optional residual connections."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            if config.residual:
                encoder_layers.append(ResidualDenseBlock(in_dim, hidden))
            else:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU(inplace=True))
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def sample_latent(self, encoded: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Return samples from a normal distribution centered at the encoded vector."""
        if not self.config.random_sampler:
            return encoded.unsqueeze(0).repeat(num_samples, 1)
        std = self.config.sampler_std
        return encoded.unsqueeze(0).repeat(num_samples, 1) + torch.randn(num_samples, encoded.size(-1), device=encoded.device) * std

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    residual: bool = False,
    random_sampler: bool = True,
    sampler_std: float = 1.0,
) -> AutoencoderNet:
    """Factory that returns a configured AutoencoderNet."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
        random_sampler=random_sampler,
        sampler_std=sampler_std,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    loss_fns: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> List[float]:
    """Train the autoencoder with a list of loss functions and an optional callback."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fns = loss_fns or [nn.MSELoss()]
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = sum(fn(recon, batch) for fn in loss_fns)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if callback:
            callback(epoch, epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "ResidualDenseBlock",
]
