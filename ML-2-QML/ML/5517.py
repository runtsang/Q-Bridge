"""
HybridSamplerQNN – Classical side of the hybrid model.

This module combines:
  * a classical sampler network (softmax over a small feed‑forward net),
  * a quantum‑inspired fully‑connected layer (implemented with PyTorch),
  * a simple regression head that mimics the EstimatorQNN,
  * an optional CNN‑based classifier that can be swapped in.

The forward pass returns a dictionary with two entries:
  * `class`: probabilities from the sampler,
  * `reg`: regression output from the estimator head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the seed modules that provide the classical analogues.
# They live in the same package as this file.
from.SamplerQNN import SamplerQNN
from.FCL import FCL
from.EstimatorQNN import EstimatorQNN

class HybridSamplerNet(nn.Module):
    """
    Hybrid sampler network that unifies classical sampling, a
    quantum‑inspired fully‑connected layer, and a regression head.
    """

    def __init__(self, use_cnn: bool = False) -> None:
        super().__init__()
        # Classical sampler producing a probability vector of shape (batch, 2)
        self.classical_sampler = SamplerQNN()

        # Quantum‑inspired fully‑connected layer that accepts a 2‑dim vector
        self.fcl = FCL()

        # Regression head identical to EstimatorQNN
        self.estimator = EstimatorQNN()

        # Optional CNN branch (mirrors the QCNet architecture)
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = self._build_cnn()
            self.fc = nn.Linear(84, 2)  # final classification head
        else:
            self.cnn = None
            self.fc = None

    def _build_cnn(self) -> nn.Module:
        """Build a lightweight CNN that mimics the QCNet architecture."""
        conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        return conv

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2) for the sampler or
            (batch, 3, H, W) for the optional CNN branch.

        Returns
        -------
        dict
            {'class': probabilities,'reg': regression_output}
        """
        # ---------- Classical sampler ----------
        probs = self.classical_sampler(x)  # (batch, 2)

        # ---------- Quantum‑inspired FCL ----------
        # The FCL expects a numpy array of shape (batch, 2)
        fcl_out = self.fcl.run(probs.detach().cpu().numpy())
        fcl_tensor = torch.tensor(fcl_out, dtype=torch.float32, device=x.device)

        # ---------- Estimator head ----------
        reg = self.estimator(fcl_tensor)  # (batch, 1)

        output = {"class": probs, "reg": reg}

        # ---------- Optional CNN branch ----------
        if self.use_cnn:
            cnn_features = self.cnn(x)          # (batch, 84)
            logits = self.fc(cnn_features)      # (batch, 2)
            output["cnn_class"] = F.softmax(logits, dim=-1)

        return output


__all__ = ["HybridSamplerNet"]
