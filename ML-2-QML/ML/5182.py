from __future__ import annotations

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Auto‑encoder components
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# ----------------------------------------------------------------------
# QCNN‑style feature extractor
# ----------------------------------------------------------------------
class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ----------------------------------------------------------------------
# Classical self‑attention helper
# ----------------------------------------------------------------------
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ----------------------------------------------------------------------
# Combined hybrid model
# ----------------------------------------------------------------------
class CombinedEstimatorQNN(nn.Module):
    """Classical hybrid model that fuses auto‑encoding, a QCNN‑style feature extractor,
    and a learnable self‑attention block before producing a scalar regression output."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Encode raw features
        self.autoencoder = Autoencoder(input_dim, latent_dim=16, hidden_dims=(64,32), dropout=0.05)
        # QCNN‑like convolutional path
        self.cnn = QCNNModel()
        # Attention parameters
        self.rotation_params = nn.Parameter(torch.randn(4*4))
        self.entangle_params = nn.Parameter(torch.randn(3))
        self.attention = ClassicalSelfAttention(embed_dim=4)
        # Final projection
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Auto‑encode
        z = self.autoencoder.encode(x)
        # 2. QCNN feature extraction
        features = self.cnn(z)
        # 3. Self‑attention
        rot_np = self.rotation_params.detach().cpu().numpy()
        ent_np = self.entangle_params.detach().cpu().numpy()
        att_out = self.attention.run(rot_np, ent_np, features.detach().cpu().numpy())
        att_tensor = torch.from_numpy(att_out).to(x.device)
        # 4. Regression head
        return self.head(att_tensor)
