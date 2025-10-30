"""Hybrid autoencoder combining classical MLP with quantum-inspired activation.

The architecture extends a standard fully‑connected autoencoder by inserting a
`QuantumActivation` layer after the encoder and before the decoder.  The
activation mimics the expectation value of a single‑qubit rotation, providing
a smooth, bounded non‑linearity that is analytically differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

def _as_tensor(data: torch.Tensor | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class QuantumActivation(torch.autograd.Function):
    """
    Differentiable layer that returns the expectation value of Pauli‑Z after a
    single‑qubit RY rotation:  ⟨Z⟩ = cos(θ).  The backward is analytic.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return torch.cos(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        return grad_output * (-torch.sin(inputs))

class HybridLayer(nn.Module):
    """
    Linear layer followed by the quantum‑inspired activation.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = QuantumActivation.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class HybridAutoencoderNet(nn.Module):
    """
    Classic autoencoder with a quantum‑inspired bottleneck.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        dim = cfg.input_dim
        for hid in cfg.hidden_dims:
            enc_layers.append(nn.Linear(dim, hid))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            dim = hid
        enc_layers.append(nn.Linear(dim, cfg.latent_dim))
        enc_layers.append(QuantumActivation.apply)  # quantum bottleneck
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        dim = cfg.latent_dim
        for hid in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(dim, hid))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            dim = hid
        dec_layers.append(nn.Linear(dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(cfg)

def train_autoencoder(
    model: nn.Module,
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "HybridLayer",
    "HybridAutoencoderNet",
    "HybridAutoencoder",
    "train_autoencoder",
]
