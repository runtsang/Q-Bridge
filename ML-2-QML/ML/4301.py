"""Hybrid sampler network that integrates classical autoencoding, quantum sampling, and quantum estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

class SamplerQNN(nn.Module):
    """Hybrid sampler network that encodes data, samples latent variables via a quantum sampler,
    decodes to output distribution, and estimates a target scalar using a quantum estimator."""
    def __init__(self,
                 quantum_sampler,
                 quantum_estimator,
                 latent_dim: int = 3,
                 input_dim: int = 2) -> None:
        super().__init__()
        self.quantum_sampler = quantum_sampler
        self.quantum_estimator = quantum_estimator

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.Tanh(),
            nn.Linear(4, latent_dim),
        )
        # Classical decoder to produce probability distribution
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Tanh(),
            nn.Linear(4, input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode input to latent representation
        latent = self.encoder(x)
        # Sample latent via quantum sampler
        q_samples = self.quantum_sampler.sample(latent)
        # Decode to output distribution
        probs = self.decoder(q_samples)
        # Estimate target scalar using quantum estimator
        raw_est = self.quantum_estimator.sample(q_samples)
        # Convert raw_est (numpy array) to torch tensor
        target = torch.tensor(raw_est, dtype=torch.float32, device=x.device)
        return probs, target

def SamplerQNNFactory(quantum_sampler, quantum_estimator) -> SamplerQNN:
    """Return a fully configured hybrid sampler network."""
    return SamplerQNN(quantum_sampler, quantum_estimator)
