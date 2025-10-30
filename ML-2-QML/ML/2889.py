"""AutoencoderGen153 – Classical implementation with optional quantum‑style block.

The model keeps the core encoder–decoder structure of the original Autoencoder
but optionally injects a QFCModel‑like feature extractor to capture
higher‑order correlations.  The design mirrors the quantum circuit
architecture from the QML seed, allowing a direct comparison of
performance and a smooth transition to a full quantum model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | List[float]) -> torch.Tensor:
    """Coerce input to a float32 tensor on the current device."""
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderGen153Config:
    """Configuration for :class:`AutoencoderGen153`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum_block: bool = False  # whether to inject QFCModel


class QFCModel(nn.Module):
    """Classical CNN + fully‑connected projection, inspired by Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


class AutoencoderGen153(nn.Module):
    """Hybrid encoder–decoder with optional quantum‑style block."""
    def __init__(self, config: AutoencoderGen153Config) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Optional quantum‑style feature extractor
        self.use_quantum_block = config.use_quantum_block
        if self.use_quantum_block:
            # The block expects a 1‑D tensor; reshape latent vectors accordingly.
            self.quantum_block = QFCModel()
        else:
            self.quantum_block = None

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        if self.use_quantum_block:
            # Reshape to match QFCModel input: (batch, 1, 28, 28) if input_dim==784
            # For general input_dim, we pad or truncate to 28x28.
            if self.config.input_dim == 784:
                latent_img = latent.view(-1, 1, 28, 28)
            else:
                # Fallback: flatten to (batch, 1, 1, input_dim)
                latent_img = latent.unsqueeze(1).unsqueeze(1)
            latent = self.quantum_block(latent_img)
        return self.decoder(latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


def AutoencoderGen153Factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum_block: bool = False,
) -> AutoencoderGen153:
    """Convenience factory mirroring the original API."""
    config = AutoencoderGen153Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_block=use_quantum_block,
    )
    return AutoencoderGen153(config)


def train_autoencoder_gen153(
    model: AutoencoderGen153,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop that returns a history of reconstruction loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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


__all__ = [
    "AutoencoderGen153",
    "AutoencoderGen153Factory",
    "AutoencoderGen153Config",
    "train_autoencoder_gen153",
    "QFCModel",
]
