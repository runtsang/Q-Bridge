"""Hybrid classical autoencoder combining self‑attention and fast estimation.

The implementation preserves the original Autoencoder API while adding a
latent‑space self‑attention layer.  A lightweight FastEstimator is also
bundled for batch evaluation with optional Gaussian shot noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Callable, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --------------------------------------------------------------------------- #
# Utility: tensor conversion
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    attention_heads: int = 2
    attention_dim: int = 4

# --------------------------------------------------------------------------- #
# Classical self‑attention layer
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Self‑attention applied to the latent vector."""

    def __init__(self, embed_dim: int, heads: int = 1, head_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(embed_dim, heads * head_dim, bias=False)
        self.k = nn.Linear(embed_dim, heads * head_dim, bias=False)
        self.v = nn.Linear(embed_dim, heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.q(x).reshape(b, self.heads, self.head_dim)
        k = self.k(x).reshape(b, self.heads, self.head_dim)
        v = self.v(x).reshape(b, self.heads, self.head_dim)
        scores = torch.einsum("bhd,bhd->bh", q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bh,bhd->bhd", attn, v).reshape(b, self.heads * self.head_dim)
        return self.out(out)

# --------------------------------------------------------------------------- #
# Autoencoder network
# --------------------------------------------------------------------------- #
class AutoencoderHybridNet(nn.Module):
    """Fully‑connected autoencoder with a self‑attention refinement of the latent code."""

    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
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

        # Self‑attention on latent
        self.attention = ClassicalSelfAttention(
            embed_dim=config.latent_dim,
            heads=config.attention_heads,
            head_dim=config.attention_dim,
        )

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def refine(self, z: torch.Tensor) -> torch.Tensor:
        return self.attention(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        z = self.refine(z)
        return self.decode(z)

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    attention_heads: int = 2,
    attention_dim: int = 4,
) -> AutoencoderHybridNet:
    """Convenience constructor mirroring the original API."""
    cfg = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
    )
    return AutoencoderHybridNet(cfg)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Standard reconstruction training with optional device placement."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)
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
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Fast estimator with optional shot noise
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Wraps a neural network for batch evaluation with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _as_tensor(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    if shots is not None:
                        scalar = float(rng.normal(scalar, max(1e-6, 1 / shots)))
                    row.append(scalar)
                results.append(row)
        return results

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderHybridConfig",
    "train_autoencoder",
    "FastEstimator",
]
