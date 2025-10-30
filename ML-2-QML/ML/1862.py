"""
Extended autoencoder implementation with residual connections,
layer‑norm, and early‑stopping.  The shared class name
`AutoencoderModel` is kept compatible with the quantum
counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


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
    """Configuration for :class:`AutoencoderModel`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64, 32)
    dropout: float = 0.1
    use_layernorm: bool = False
    early_stop_patience: int | None = 10


class AutoencoderModel(nn.Module):
    """
    Residual autoencoder with optional layer‑norm and early‑stopping
    training support.  The encoder/decoder are mirrored and each
    hidden layer optionally contains a residual skip connection.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_block(
            config.input_dim, config.hidden_dims, config.latent_dim
        )
        self.decoder = self._build_block(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            reverse=True,
        )

    def _build_block(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        reverse: bool = False,
    ) -> nn.Module:
        layers: List[nn.Module] = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if self.config.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU(inplace=True))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(p=self.config.dropout))
            # residual connection if dimensions match
            if in_dim == hidden:
                layers.append(nn.Identity())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the latent."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    @staticmethod
    def _train(
        model: "AutoencoderModel",
        data: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
        early_stop_patience: int | None = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        history: List[float] = []
        best_loss = float("inf")
        patience_counter = 0

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
            # Early‑stopping check
            if early_stop_patience is not None:
                if epoch_loss < best_loss - 1e-5:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
        return history

    @classmethod
    def train(
        cls,
        config: AutoencoderConfig,
        data: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> tuple["AutoencoderModel", List[float]]:
        """Convenience wrapper returning a trained model and loss history."""
        model = cls(config)
        history = cls._train(
            model,
            data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            early_stop_patience=config.early_stop_patience,
        )
        return model, history


__all__ = [
    "AutoencoderConfig",
    "AutoencoderModel",
    "AutoencoderModel.train",
]
