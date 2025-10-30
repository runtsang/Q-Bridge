import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNN(nn.Module):
    """
    Hybrid classical‑to‑quantum sampler network.

    Architecture:
    - Feature extractor: two‑layer MLP mapping an arbitrary input vector
      into a 2‑dimensional latent space.
    - The latent vector is interpreted as the input parameters of a
      two‑qubit variational sampler circuit.
    - The network outputs a probability distribution over the 4 basis states
      of a 2‑qubit system.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # latent 2‑dim vector
        )
        self.decoder = nn.Linear(2, 4)  # 4 basis states for 2 qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over 4 basis states.
        """
        latent = self.encoder(x)                     # (batch, 2)
        logits = self.decoder(latent)                # (batch, 4)
        probs = F.softmax(logits, dim=-1)            # (batch, 4)
        return probs

    def sample_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the latent vector that will be used as quantum circuit parameters.
        """
        return self.encoder(x)                       # (batch, 2)

__all__ = ["SamplerQNN"]
