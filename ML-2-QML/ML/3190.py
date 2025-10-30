"""Classical self‑attention implementation compatible with the quantum API."""
import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """
    Classical self‑attention using NumPy‑style parameters.
    Parameters are passed as NumPy arrays to mirror the quantum interface.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: dimensionality of the input embeddings.
        """
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention scores and weighted sum.

        Args:
            rotation_params: shape (embed_dim, embed_dim) – linear map for queries.
            entangle_params: shape (embed_dim, embed_dim) – linear map for keys.
            inputs: shape (seq_len, embed_dim) – input embeddings.

        Returns:
            np.ndarray of shape (seq_len, embed_dim) – attended representations.
        """
        # Validate shapes
        if rotation_params.shape!= (self.embed_dim, self.embed_dim):
            raise ValueError(
                f"rotation_params must have shape "
                f"({self.embed_dim}, {self.embed_dim}), got {rotation_params.shape}"
            )
        if entangle_params.shape!= (self.embed_dim, self.embed_dim):
            raise ValueError(
                f"entangle_params must have shape "
                f"({self.embed_dim}, {self.embed_dim}), got {entangle_params.shape}"
            )

        # Convert to torch tensors
        q = torch.from_numpy(inputs @ rotation_params).float()
        k = torch.from_numpy(inputs @ entangle_params).float()
        v = torch.from_numpy(inputs).float()

        # Scaled dot‑product attention
        scores = F.softmax(
            q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1
        )
        out = scores @ v
        return out.detach().cpu().numpy()
