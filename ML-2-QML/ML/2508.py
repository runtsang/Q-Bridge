import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridFCLAttention(nn.Module):
    """
    Classical hybrid fully‑connected + self‑attention layer.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    embed_dim : int, default 4
        Size of the intermediate embedding and number of qubits used in the
        quantum‑style attention block.

    The layer accepts a flat iterable ``thetas`` containing:

    * ``n_features * embed_dim`` weights for the linear map
    * ``embed_dim`` biases for the linear map
    * ``embed_dim * 3`` rotation parameters for the attention
    * ``(embed_dim - 1) * embed_dim`` entanglement parameters

    The forward pass first applies the linear map and then a classical
    self‑attention block using the supplied rotation and entanglement
    parameters.  The interface mirrors the original FCL example while
    exposing a richer parameter space that can be jointly optimized.
    """
    def __init__(self, n_features: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.linear = nn.Linear(n_features, embed_dim, bias=True)

    def run(self, thetas: Iterable[float], inputs: np.ndarray) -> np.ndarray:
        # Parse the theta vector
        thetas = np.asarray(thetas, dtype=np.float32)
        w_len = self.n_features * self.embed_dim
        weight = thetas[:w_len].reshape(self.embed_dim, self.n_features)
        bias = thetas[w_len:w_len + self.embed_dim]

        start = w_len + self.embed_dim
        rotation_len = self.embed_dim * 3
        rotation_params = thetas[start:start + rotation_len].reshape(self.embed_dim, 3)
        entangle_len = (self.embed_dim - 1) * self.embed_dim
        entangle_params = thetas[start + rotation_len:start + rotation_len + entangle_len].reshape(self.embed_dim - 1, self.embed_dim)

        # Update linear parameters
        self.linear.weight.data = torch.tensor(weight, dtype=torch.float32)
        self.linear.bias.data = torch.tensor(bias, dtype=torch.float32)

        # Forward pass
        x = torch.as_tensor(inputs, dtype=torch.float32)
        linear_out = self.linear(x)  # shape (batch, embed_dim)

        # Classical self‑attention
        query = linear_out @ rotation_params.T
        key = linear_out @ entangle_params.T
        value = linear_out
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ value

        return attn_out.detach().numpy()

def FCL(n_features: int = 1, embed_dim: int = 4) -> HybridFCLAttention:
    """
    Factory that returns a classical HybridFCLAttention instance.
    """
    return HybridFCLAttention(n_features, embed_dim)

__all__ = ["HybridFCLAttention", "FCL"]
