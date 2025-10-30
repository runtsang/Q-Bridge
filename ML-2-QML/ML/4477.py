'''python
# Classical hybrid model combining convolution, auto‑encoder, and linear head.
# The auto‑encoder reduces the 784‑dimensional feature vector from the
# 2‑D convolution before classification, providing a compact latent
# representation that complements the quantum‑style feature extraction
# in the QML counterpart.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightweight fully‑connected auto‑encoder
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# Main hybrid model
class QuanvolutionHybrid(nn.Module):
    def __init__(self,
                 conv_out_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 autoencoder_cfg: dict | None = None,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=conv_kernel, stride=conv_stride)

        cfg = autoencoder_cfg or {
            "input_dim": conv_out_channels * 14 * 14,
            "latent_dim": 32,
            "hidden_dims": (128, 64),
            "dropout": 0.1,
        }
        self.autoencoder = AutoencoderNet(**cfg)
        self.classifier = nn.Linear(cfg["latent_dim"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)                # (batch, C, 14, 14)
        flat = conv_out.view(x.size(0), -1)    # (batch, 784)
        encoded = self.autoencoder.encode(flat)  # (batch, latent_dim)
        logits = self.classifier(encoded)      # (batch, num_classes)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
'''
