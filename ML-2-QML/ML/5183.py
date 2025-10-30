from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the lightweight helpers that mirror the quantum implementation
from.SelfAttention import SelfAttention
from.SamplerQNN import SamplerQNN
# Re‑use the data generation and dataset class from the original anchor
from.QuantumRegression import generate_superposition_data, RegressionDataset

class HybridRegressionModel(nn.Module):
    """Hybrid classical regression model that fuses self‑attention and a sampler."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.attn = SelfAttention()
        self.sampler = SamplerQNN()
        # Feature expansion: original + attention output
        expanded_dim = num_features + num_features
        self.net = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, num_features).
        """
        # --- Classical self‑attention (placeholder parameters) ---
        rotation_params = np.linspace(0, np.pi, 12, dtype=np.float32)  # 4 wires × 3 angles
        entangle_params = np.zeros(3, dtype=np.float32)
        attn_output = self.attn.run(rotation_params, entangle_params, x.cpu().numpy())
        attn_tensor = torch.tensor(attn_output, dtype=torch.float32, device=x.device)

        # Concatenate original features with attention output
        combined = torch.cat([x, attn_tensor], dim=1)

        # --- Sampler‑based weighting ---
        probs = self.sampler(combined)          # shape (batch, 2)
        weight = probs[:, 0].unsqueeze(-1)      # use first class probability as weight
        weighted = combined * weight

        # --- Feed‑forward regression head ---
        return self.net(weighted).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
