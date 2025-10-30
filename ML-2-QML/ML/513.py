import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Classical self‑attention module that mirrors the quantum interface.

    Added features:
        * configurable temperature for the softmax
        * optional dropout on the attention map
        * a linear projection after the weighted sum
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        # projection layer to keep the interface identical to the seed
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.embed_dim ** 0.5 * self.temperature)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention using the provided parameters.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters reshaped to a ``(embed_dim, embed_dim)`` matrix for the query projection.
        entangle_params : np.ndarray
            Parameters reshaped to a ``(embed_dim, embed_dim)`` matrix for the key projection.
        inputs : np.ndarray
            ``(batch, embed_dim)`` input tensor.

        Returns
        -------
        np.ndarray
            The attended representation with the same shape as ``inputs``.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        query = torch.matmul(inputs_t,
                             torch.as_tensor(rotation_params.reshape(self.embed_dim, -1),
                                             dtype=torch.float32))
        key   = torch.matmul(inputs_t,
                             torch.as_tensor(entangle_params.reshape(self.embed_dim, -1),
                                             dtype=torch.float32))
        value = inputs_t
        out = self._attention(query, key, value)
        out = self.proj(out)
        return out.detach().cpu().numpy()

    # ``forward`` is kept for compatibility with ``nn.Module`` usage
    forward = run
