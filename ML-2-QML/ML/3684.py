from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# ----- Classical Autoencoder -----
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ----- Hybrid Classifier -----
class HybridAutoencoderClassifier(nn.Module):
    """
    Classical CNN + Autoencoder + dense head for binary classification.
    The autoencoder compresses the flattened image.
    The dense head mimics a quantum expectation layer.
    """
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 autoencoder_cfg: AutoencoderConfig,
                 hidden_features: int = 84) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(autoencoder_cfg)

        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.5),
            nn.Flatten()
        )

        # Infer CNN feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            cnn_out = self.cnn(dummy)
        cnn_feat_dim = cnn_out.shape[1]

        self.fc1 = nn.Linear(cnn_feat_dim + autoencoder_cfg.latent_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)
        self.shift = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        cnn_feat = self.cnn(x)
        flat = x.view(x.size(0), -1)
        latents = self.autoencoder.encode(flat)
        combined = torch.cat([cnn_feat, latents], dim=1)
        h = F.relu(self.fc1(combined))
        logits = self.fc2(h)
        prob = torch.sigmoid(logits + self.shift)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["AutoencoderConfig", "AutoencoderNet", "HybridAutoencoderClassifier"]
