import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention:
    """Simple self‑attention block that mimics the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class UnifiedSamplerEstimatorAttention(nn.Module):
    """
    Classical hybrid network that combines:
    1. A lightweight sampler network (softmax over 2 outputs).
    2. A small regression network that mimics the quantum estimator.
    3. A self‑attention block that produces a 4‑dimensional representation.
    4. A final dense head that aggregates the three sources of information
       and outputs binary class probabilities.
    """
    def __init__(self,
                 sampler_hidden: int = 4,
                 estimator_hidden: int = 8,
                 attention_embed_dim: int = 4):
        super().__init__()

        # Classical sampler
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2)
        )

        # Classical estimator (regression head)
        self.estimator = nn.Sequential(
            nn.Linear(2, estimator_hidden),
            nn.Tanh(),
            nn.Linear(estimator_hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

        # Self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=attention_embed_dim)

        # Final classification head
        # 2 (sampler) + 1 (estimator) + attention_embed_dim (attention)
        self.final = nn.Linear(2 + 1 + attention_embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Binary class probabilities of shape (batch, 2).
        """
        # Sampler output (softmax)
        sampler_out = F.softmax(self.sampler(x), dim=-1)

        # Estimator output (scalar)
        estimator_out = self.estimator(x).squeeze(-1)

        # Self‑attention: use zero parameters for a deterministic baseline
        rotation_params = np.zeros(self.attention.embed_dim * 3)
        entangle_params = np.zeros(self.attention.embed_dim - 1)
        attention_out = self.attention.run(rotation_params, entangle_params, x.numpy())

        # Convert attention output to torch tensor
        attention_out = torch.tensor(attention_out, dtype=torch.float32)

        # Concatenate all signals
        combined = torch.cat([sampler_out,
                              estimator_out.unsqueeze(-1),
                              attention_out], dim=-1)

        logits = self.final(combined)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["UnifiedSamplerEstimatorAttention"]
