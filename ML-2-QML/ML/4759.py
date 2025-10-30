import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

# Classical self‑attention layer
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, embed_dim)
        query = inputs @ self.rotation_params
        key   = inputs @ self.entangle_params
        scores = nn.functional.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

# Auto‑encoder configuration
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

# Simple MLP auto‑encoder
class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# Hybrid network that injects self‑attention into the latent space
class SelfAttentionAutoEncoderAttentionNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(config)
        self.attention = ClassicalSelfAttention(config.latent_dim)

    def encode(self, inputs: Tensor) -> Tensor:
        z = self.autoencoder.encode(inputs)
        return self.attention(z)

    def decode(self, latents: Tensor) -> Tensor:
        return self.autoencoder.decode(latents)

    def forward(self, inputs: Tensor) -> Tensor:
        z = self.autoencoder.encode(inputs)
        z = self.attention(z)
        return self.autoencoder.decode(z)

# Convenience factory
def SelfAttentionAutoEncoderAttention(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> SelfAttentionAutoEncoderAttentionNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return SelfAttentionAutoEncoderAttentionNet(cfg)

# Simple training loop (kept for reference)
def train_autoencoder(
    model: nn.Module,
    data: Tensor,
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: Iterable[float] | Tensor) -> Tensor:
    if isinstance(data, Tensor):
        return data
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "ClassicalSelfAttention",
    "AutoencoderConfig",
    "AutoencoderNet",
    "SelfAttentionAutoEncoderAttentionNet",
    "SelfAttentionAutoEncoderAttention",
    "train_autoencoder",
]
