import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = False
    activation: nn.Module = nn.ReLU

class AutoencoderGen308(nn.Module):
    """Configurable MLP autoencoder with optional batch‑norm and early‑stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            batchnorm=config.batchnorm,
            activation=config.activation,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=config.hidden_dims[::-1],
            dropout=config.dropout,
            batchnorm=config.batchnorm,
            activation=config.activation,
        )

    def _build_mlp(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        batchnorm: bool,
        activation: nn.Module,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder(
    model: AutoencoderGen308,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stop: int | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_loss = float("inf")
    patience = early_stop or 0
    counter = 0

    for epoch in range(epochs):
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
        if scheduler is not None:
            scheduler.step()
        if early_stop is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stop:
                    break
    return history

__all__ = ["AutoencoderGen308", "AutoencoderConfig", "train_autoencoder"]
