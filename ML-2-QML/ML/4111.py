from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter producing four feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        return self.conv(x).view(x.size(0), -1)  # (batch, 4*14*14)

class ClassicalSelfAttention(nn.Module):
    """Self‑attention block with the same interface as the quantum version."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor,
                rot_params: torch.Tensor,
                ent_params: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, embed_dim)
        query = self.query_proj(inputs @ rot_params.reshape(self.embed_dim, -1))
        key   = self.key_proj(inputs @ ent_params.reshape(self.embed_dim, -1))
        value = self.value_proj(inputs)
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder used to compress the attention output."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            dim = h
        encoder_layers.append(nn.Linear(dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            dim = h
        decoder_layers.append(nn.Linear(dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QuanvolutionAutoencoderClassifier(nn.Module):
    """Hybrid network that chains a quanvolution filter, classical self‑attention,
    a fully‑connected auto‑encoder and a linear head for classification."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter   = QuanvolutionFilter()
        self.attn      = ClassicalSelfAttention(embed_dim=4)
        self.autoenc   = AutoencoderNet(input_dim=4 * 14 * 14,
                                        latent_dim=64,
                                        hidden_dims=(128, 64),
                                        dropout=0.1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.qfilter(x)  # (batch, 4*14*14)
        seq = features.view(features.size(0), 14 * 14, 4)  # (batch, seq_len, embed_dim)
        # placeholder rotation/entanglement params for the attention block
        rot_params = torch.rand(4, 4, device=x.device)
        ent_params = torch.rand(4, 4, device=x.device)
        attn_out = self.attn(seq, rot_params, ent_params)  # (batch, seq_len, embed_dim)
        flat = attn_out.reshape(attn_out.size(0), -1)  # (batch, seq_len*embed_dim)
        encoded = self.autoenc.encode(flat)  # (batch, latent_dim)
        logits = self.classifier(encoded)    # (batch, num_classes)
        return F.log_softmax(logits, dim=-1)

# Backwards‑compatibility alias used by the original test harness
QuanvolutionClassifier = QuanvolutionAutoencoderClassifier

__all__ = ["QuanvolutionFilter",
           "ClassicalSelfAttention",
           "AutoencoderNet",
           "QuanvolutionAutoencoderClassifier",
           "QuanvolutionClassifier"]
