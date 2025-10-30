"""Hybrid autoencoder combining classical MLP/Transformer encoder-decoder with a quantum latent module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import quantum latent module
from.quantum_latent import QuantumLatentQNN, build_latent_circuit


@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_transformer: bool = False
    transformer_params: Optional[dict] = None  # {"num_heads":4, "ffn_dim":128, "num_blocks":2}
    # Quantum parameters
    q_circuit_builder: Callable[[int], nn.Module] | None = None
    q_device: str | None = None
    q_batch_size: int = 1


# --------------------------------------------------------------------------- #
# 1. Classical encoder/decoder (MLP or Transformer)
# --------------------------------------------------------------------------- #
class TransformerBlockClassical(nn.Module):
    """Simple transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class ClassicalEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        if config.use_transformer:
            embed_dim = config.input_dim
            num_heads = config.transformer_params.get("num_heads", 4)
            ffn_dim = config.transformer_params.get("ffn_dim", 128)
            num_blocks = config.transformer_params.get("num_blocks", 2)
            self.encoder = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, config.dropout) for _ in range(num_blocks)]
            )
            self.final = nn.Linear(embed_dim, config.latent_dim)
        else:
            layers = []
            in_dim = config.input_dim
            for h in config.hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x


class ClassicalDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        if config.use_transformer:
            embed_dim = config.latent_dim
            num_heads = config.transformer_params.get("num_heads", 4)
            ffn_dim = config.transformer_params.get("ffn_dim", 128)
            num_blocks = config.transformer_params.get("num_blocks", 2)
            self.decoder = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, config.dropout) for _ in range(num_blocks)]
            )
            self.final = nn.Linear(embed_dim, config.input_dim)
        else:
            layers = []
            in_dim = config.latent_dim
            for h in reversed(config.hidden_dims):
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        if hasattr(self, "final"):
            x = self.final(x)
        return x


# --------------------------------------------------------------------------- #
# 2. Quantum latent module
# --------------------------------------------------------------------------- #
class QuantumLatentModule(nn.Module):
    """Wraps a quantum neural network that transforms a classical latent vector."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        if config.q_circuit_builder is None:
            raise ValueError("Quantum circuit builder must be provided.")
        self.qnn = QuantumLatentQNN(
            circuit_builder=config.q_circuit_builder,
            latent_dim=config.latent_dim,
            q_device=config.q_device,
            q_batch_size=config.q_batch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)


# --------------------------------------------------------------------------- #
# 3. Hybrid autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderGen190(nn.Module):
    """Hybrid autoencoder combining classical encoder/decoder with quantum latent transform."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = ClassicalEncoder(config)
        self.quantum = QuantumLatentModule(config)
        self.decoder = ClassicalDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent_q = self.quantum(latent)
        recon = self.decoder(latent_q)
        return recon


# --------------------------------------------------------------------------- #
# 4. Training helper
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_autoencoder(
    model: AutoencoderGen190,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
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


__all__ = ["AutoencoderGen190", "AutoencoderConfig", "train_autoencoder"]
