"""
Classical sampler network with enhanced depth, regularisation, and sampling utilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def SamplerQNN() -> nn.Module:
    """
    Returns an instance of the updated SamplerQNN__gen088 module.
    The network now contains:
    - Two hidden layers (8 and 4 units) with ReLU activations.
    - Batch Normalisation after the first hidden layer.
    - Dropout (p=0.1) to mitigate over‑fitting.
    - Xavier weight initialization for better convergence.
    - ``sample`` method to draw discrete samples from the output distribution.
    - ``kl_divergence`` helper for KL loss against a target distribution.
    """
    class SamplerQNN__gen088(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.BatchNorm1d(8),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(4, 2),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Compute softmax probabilities over the two output classes.
            """
            return F.softmax(self.net(inputs), dim=-1)

        def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
            """
            Draw discrete samples from the output distribution.
            Parameters
            ----------
            inputs : torch.Tensor
                Input tensor of shape (batch, 2) or (2,).
            num_samples : int
                Number of samples to draw for each batch entry.
            Returns
            -------
            torch.Tensor
                Tensor of sampled class indices of shape (batch, num_samples).
            """
            probs = self.forward(inputs).detach().cpu().numpy()
            # Ensure probs shape is (batch, 2)
            if probs.ndim == 1:
                probs = probs[np.newaxis, :]
            samples = np.array([np.random.choice(2, size=num_samples, p=p) for p in probs])
            return torch.tensor(samples)

        def kl_divergence(self, target: torch.Tensor) -> torch.Tensor:
            """
            Compute KL divergence between the network output and a target distribution.
            Parameters
            ----------
            target : torch.Tensor
                Target probability distribution of shape (batch, 2).
            Returns
            -------
            torch.Tensor
                KL divergence for each batch entry.
            """
            probs = self.forward(target)
            log_probs = torch.log(probs + 1e-12)
            log_target = torch.log(target + 1e-12)
            return torch.sum(probs * (log_probs - log_target), dim=-1)

    return SamplerQNN__gen088()


__all__ = ["SamplerQNN"]
