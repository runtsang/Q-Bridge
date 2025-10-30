"""Enhanced autoencoder with hybrid support and flexible training pipeline."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderConfig:
    """Configuration for :class:`AutoencoderModel`."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderModel(nn.Module):
    """Fully‑connected autoencoder with optional hybrid interface."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=tuple(reversed(config.hidden_dims)),
            dropout=config.dropout,
        )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int,
                   hidden_dims: Tuple[int,...],
                   dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Hybrid helpers
    # ------------------------------------------------------------------
    def latent_dim(self) -> int:
        """Return the dimensionality of the latent space."""
        return self.config.latent_dim

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> List[float]:
        """Train the autoencoder with optional partial freezing."""
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(device)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> "AutoencoderModel":
        return cls(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

__all__ = ["AutoencoderModel", "AutoencoderConfig"]
