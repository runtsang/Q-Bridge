"""Hybrid classical sampler network combining autoencoder encoding and softmax output.

The network mirrors the quantum sampler but uses a lightweight
autoencoder encoder to reduce dimensionality before the final
classification layer.  It is fully compatible with the original
SamplerQNN anchor while providing richer feature extraction.

The architecture:
    encoder (Autoencoder encoder) -> linear -> softmax

The module can be used directly in standard PyTorch pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    def __init__(self, input_dim: int = 2, latent_dim: int = 3,
                 hidden_dims: tuple[int, int] = (128, 64)):
        super().__init__()
        # Encoder part from the AutoencoderNet
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classifier
        self.classifier = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        logits = self.classifier(z)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridSamplerQNN"]
