from __future__ import annotations

import torch
from torch import nn
from Autoencoder import Autoencoder
from Conv import Conv

class HybridRegression(nn.Module):
    """Classical hybrid regression model.

    Consists of an auto‑encoder for dimensionality reduction, a
    quantum‑inspired convolution filter, a small MLP mimicking a
    variational layer, and a linear head producing a scalar
    regression output.
    """
    def __init__(self, num_features: int, latent_dim: int = 32):
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=num_features,
            latent_dim=latent_dim,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.conv = Conv()
        self.quantum_layer = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.autoencoder.encode(x)  # (B, latent_dim)
        conv_outs = []
        for sample in encoded.view(-1, 2, 2):
            conv_out = self.conv.run(sample.cpu().numpy())
            conv_outs.append(conv_out)
        conv_tensor = torch.tensor(conv_outs, dtype=x.dtype, device=x.device).unsqueeze(-1)
        q_out = self.quantum_layer(conv_tensor)
        return self.head(q_out).squeeze(-1)
