"""Combined classical self‑attention and feed‑forward regressor."""
import numpy as np
import torch
from torch import nn

class ClassicalSelfAttention:
    """Self‑attention layer that mirrors the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Linear projections
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        # Attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class EstimatorNN(nn.Module):
    """Tiny fully‑connected regression network."""
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class CombinedAttentionEstimator:
    """
    Classical self‑attention followed by a regression head.
    The interface matches the pure attention and pure estimator modules,
    enabling seamless hybrid experiments.
    """
    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8):
        self.attention = ClassicalSelfAttention(embed_dim)
        # The attention output dimension equals the input dimension (embed_dim)
        self.regressor = EstimatorNN(input_dim=embed_dim, hidden_dim=hidden_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Forward pass through attention and regression.
        """
        attn_out = self.attention.run(rotation_params, entangle_params, inputs)
        x = torch.as_tensor(attn_out, dtype=torch.float32, device=device)
        pred = self.regressor(x)
        return pred.detach().cpu().numpy()

__all__ = ["CombinedAttentionEstimator"]
