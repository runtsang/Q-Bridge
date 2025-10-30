"""Enhanced fully-connected autoencoder with flexible architecture and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    """Configuration values for :class:`AutoencoderModel`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: nn.Module = nn.ReLU()
    bias: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden, bias=config.bias))
            encoder_layers.append(config.activation)
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim, bias=config.bias))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden, bias=config.bias))
            decoder_layers.append(config.activation)
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim, bias=config.bias))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


class AutoencoderModel(nn.Module):
    """Encapsulates an autoencoder with training, evaluation and early‑stopping."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.net = AutoencoderNet(config)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net.decode(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = False,
    ) -> List[float]:
        """Train the autoencoder and return a list of epoch‑wise training losses."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch + 1:03d} – loss: {epoch_loss:.6f}")

            # Early‑stopping logic
            if early_stopping_patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        return history

    def evaluate(self, data: torch.Tensor, device: Optional[torch.device] = None) -> float:
        """Return the mean‑squared‑error on the provided data."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            data = _as_tensor(data).to(device)
            reconstruction = self(data)
            mse = nn.functional.mse_loss(reconstruction, data).item()
        self.train()
        return mse

    @classmethod
    def from_config(cls, config: AutoencoderConfig) -> "AutoencoderModel":
        """Convenience constructor."""
        return cls(config)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "AutoencoderModel",
]
