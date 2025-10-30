import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """
    Classical self‑attention module that mirrors the original seed's
    interface while exposing trainable weight matrices.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query/key/value space.
    Methods
    -------
    forward(inputs, rotation_params, entangle_params)
        Computes the attention output using the supplied weight matrices.
    run(rotation_params, entangle_params, inputs)
        Convenience wrapper that accepts plain numpy arrays.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Trainable weight matrices that can be overwritten by external params
        self.query_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_weight   = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        # Load external rotation/entangle parameters into the weight matrices
        self.query_weight.data = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        self.key_weight.data   = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()

        query = inputs @ self.query_weight
        key   = inputs @ self.key_weight
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        value = inputs
        return scores @ value

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        inputs_t = torch.from_numpy(inputs).float()
        out = self.forward(inputs_t, rotation_params, entangle_params)
        return out.detach().numpy()

__all__ = ["SelfAttention"]
