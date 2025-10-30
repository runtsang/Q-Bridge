from __future__ import annotations
import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """
    Hybrid self‑attention that fuses classical attention, a QCNN‑style
    feed‑forward network, a fully‑connected layer and an EstimatorQNN‑style regressor.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear maps for query/key/value derived from rotation and entangle params
        self.query_weight = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_weight = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_weight = nn.Linear(embed_dim, embed_dim, bias=False)

        # QCNN‑style feed‑forward
        self.cnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # Fully‑connected layer (FCL)
        self.fcl = nn.Linear(hidden_dim, 1)

        # EstimatorQNN‑style regressor
        self.regressor = nn.Sequential(
            nn.Linear(1, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 1),
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim) used as weight matrix for queries.
        entangle_params : np.ndarray
            Shape (embed_dim, embed_dim) used as weight matrix for keys.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            The sigmoid‑activated output of the hybrid network.
        """
        # Apply linear maps derived from the provided parameters
        query = torch.as_tensor(inputs @ rotation_params, dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params, dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        attn_output = scores @ value

        # QCNN‑style processing
        x = self.cnn(attn_output)

        # FCL processing
        x = torch.tanh(self.fcl(x))

        # EstimatorQNN processing
        x = self.regressor(x)

        return torch.sigmoid(x).detach().numpy()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.regressor(torch.tanh(self.fcl(self.cnn(inputs)))))

__all__ = ["HybridSelfAttention"]
