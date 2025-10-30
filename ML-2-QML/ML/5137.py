import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class _IdentityQuantumLayer(nn.Module):
    """Fallback quantum layer that simply forwards its input."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder: classical encoder + quantum latent layer + classical decoder."""
    def __init__(self, config: AutoencoderConfig, quantum_layer: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = self._build_encoder(config)
        self.quantum_layer = quantum_layer if quantum_layer is not None else _IdentityQuantumLayer()
        self.decoder = self._build_decoder(config)

    def _build_encoder(self, config: AutoencoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self, config: AutoencoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_encode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.quantum_layer(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(x)
        qlatent = self.quantum_encode(latent)
        return self.decode(qlatent)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: nn.Module | None = None,
) -> HybridAutoencoder:
    """Factory that returns a hybrid autoencoder with an optional quantum latent layer."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(cfg, quantum_layer=quantum_layer)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["Autoencoder", "HybridAutoencoder", "AutoencoderConfig", "train_autoencoder"]
