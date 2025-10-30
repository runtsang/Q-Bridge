import numpy as np
import torch
from torch import nn

class SelfAttentionEnhanced:
    """
    Classical self‑attention module with optional linear projection and
    configurable temperature for the soft‑max.
    """
    def __init__(self, embed_dim: int, add_linear: bool = False, temperature: float = 1.0):
        self.embed_dim = embed_dim
        self.add_linear = add_linear
        self.temperature = temperature
        if add_linear:
            self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query projection, reshaped to (embed_dim, -1).
        entangle_params : np.ndarray
            Parameters for the key projection, reshaped to (embed_dim, -1).
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, embed_dim).
        """
        if self.add_linear:
            inputs = self.proj(torch.as_tensor(inputs, dtype=torch.float32)).detach().numpy()

        # Reshape parameter matrices
        q_mat = rotation_params.reshape(self.embed_dim, -1)
        k_mat = entangle_params.reshape(self.embed_dim, -1)

        # Compute query, key, value
        query = torch.as_tensor(inputs @ q_mat, dtype=torch.float32)
        key   = torch.as_tensor(inputs @ k_mat, dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Attention scores with temperature scaling
        scores = torch.softmax(query @ key.T / (np.sqrt(self.embed_dim) * self.temperature), dim=-1)
        return (scores @ value).numpy()
