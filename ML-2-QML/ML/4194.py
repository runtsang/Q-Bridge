"""Classical hybrid network that fuses an autoencoder with a quantum‑inspired head.

The network first compresses the input via a fully‑connected autoencoder, processes the latent vector through dense layers, and finally produces a binary probability using a parameterised fully‑connected layer that mimics a quantum expectation value. This design preserves the CNN‑like feature extraction while providing a differentiable, classical surrogate for the quantum head, enabling efficient training on classical hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical autoencoder and a quantum‑inspired fully‑connected layer
from Autoencoder import Autoencoder  # from reference 3
from FCL import FCL  # from reference 2

class HybridAutoencoderQCNet(nn.Module):
    """Hybrid network that fuses a fully‑connected autoencoder with a quantum‑inspired head.

    The network follows a three‑stage pipeline:
    1. Autoencoder compresses the input into a low‑dimensional latent space.
    2. Dense layers process the latent vector.
    3. A parameterised fully‑connected layer (FCL) replaces the quantum expectation head,
       providing a differentiable sigmoid‑like output while mimicking the behaviour of a
       quantum circuit.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()

        # Stage 1 – classical autoencoder
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Stage 2 – dense processing
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, 1),
        )

        # Stage 3 – quantum‑inspired final head
        self.head = FCL()  # returns a module with a `run` method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2) containing class probabilities.
        """
        # Encode
        latent = self.autoencoder(x)

        # Dense layers
        logits = self.fc(latent).squeeze(-1)  # shape (batch,)

        # Quantum‑inspired head
        probs = []
        for logit in logits.tolist():
            # FCL expects an iterable of angles; we supply the scalar logit
            probs.append(self.head.run([logit])[0])

        probs = torch.tensor(probs, device=x.device, dtype=torch.float32)

        # Return probabilities for both classes
        return torch.cat((probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)), dim=-1)

__all__ = ["HybridAutoencoderQCNet"]
