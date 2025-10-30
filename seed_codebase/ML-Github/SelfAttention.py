"""Classical self-attention helper mirroring the quantum circuit interface."""

from __future__ import annotations

import numpy as np
import torch


def SelfAttention():
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)


__all__ = ["SelfAttention"]
