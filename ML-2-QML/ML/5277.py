"""Hybrid autoencoder combining MLP, self‑attention, fraud‑detection style layers and a classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Classical self‑attention helper
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs

# Fraud‑detection style linear layer
class FraudLayer(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        self.linear.weight.data = weight
        self.linear.bias.data = bias
        self.scale = scale
        self.shift = shift
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.activation(self.linear(x)) * self.scale + self.shift)

@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    attention_dim: int = 4
    fraud_layers: int = 2

class HybridAutoencoder(nn.Module):
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Self‑attention
        self.attention = ClassicalSelfAttention(config.attention_dim)

        # Fraud‑detection style layers
        self.fraud_layers = nn.ModuleList()
        for _ in range(config.fraud_layers):
            weight = torch.randn(2, 2)
            bias = torch.randn(2)
            scale = torch.randn(2)
            shift = torch.randn(2)
            self.fraud_layers.append(FraudLayer(weight, bias, scale, shift))

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Classifier head
        self.classifier = nn.Linear(config.input_dim, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        # Attention on latent space
        rotation = torch.randn(self.attention.attention_dim, self.attention.attention_dim)
        entangle = torch.randn(self.attention.attention_dim, self.attention.attention_dim)
        z = self.attention(rotation, entangle, z)
        # Fraud layers
        for layer in self.fraud_layers:
            z = layer(z)
        recon = self.decode(z)
        logits = self.classifier(recon)
        return recon, logits

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    attention_dim: int = 4,
    fraud_layers: int = 2,
) -> HybridAutoencoder:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_dim=attention_dim,
        fraud_layers=fraud_layers,
    )
    return HybridAutoencoder(cfg)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
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
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "train_hybrid_autoencoder"]
