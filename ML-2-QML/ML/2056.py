import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input data into a float32 torch.Tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderGen265Config:
    """Configuration for the fully‑connected autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class AutoencoderGen265(nn.Module):
    """Standard fully‑connected autoencoder with configurable depth."""
    def __init__(self, config: AutoencoderGen265Config) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            in_features=config.input_dim,
            out_features=config.latent_dim,
            hidden_sizes=config.hidden_dims,
            dropout=config.dropout,
        )
        self.decoder = self._build_mlp(
            in_features=config.latent_dim,
            out_features=config.input_dim,
            hidden_sizes=config.hidden_dims[::-1],
            dropout=config.dropout,
        )

    @staticmethod
    def _build_mlp(in_features: int, out_features: int,
                   hidden_sizes: Tuple[int,...], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        last = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_features))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def AutoencoderGen265_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderGen265:
    """Convenience constructor mirroring the original API."""
    config = AutoencoderGen265Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderGen265(config)

def train_autoencoder(
    model: AutoencoderGen265,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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
    "AutoencoderGen265",
    "AutoencoderGen265Config",
    "AutoencoderGen265_factory",
    "train_autoencoder",
]
