"""Hybrid fraud‑detection model – classical implementation.

This module defines :class:`FraudDetectionHybridModel` that
* extracts image features with a small CNN (inspired by Quantum‑NAT),
* projects them to a 64‑dimensional space,
* optionally forwards the representation to a quantum submodule,
* and finally maps the result to the target class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Callable


class FraudDetectionHybridModel(nn.Module):
    """
    Classical fraud‑detection backbone with optional quantum augmentation.

    Parameters
    ----------
    num_classes : int, default=2
        Number of output classes.
    use_quantum : bool, default=False
        If ``True`` the model expects a quantum submodule to be set via
        :meth:`set_quantum_layer`.
    quantum_layer : Optional[Callable], default=None
        A callable that accepts a tensor of shape ``(batch, 64)`` and returns a
        tensor of the same shape.  Typically a quantum module from the
        :mod:`qml` package.
    """

    def __init__(
        self,
        num_classes: int = 2,
        use_quantum: bool = False,
        quantum_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.out = nn.Linear(64, num_classes)
        self.use_quantum = use_quantum
        self.quantum_layer = quantum_layer

    def set_quantum_layer(self, quantum_layer: Callable) -> None:
        """Attach a quantum submodule to the classifier."""
        self.quantum_layer = quantum_layer
        self.use_quantum = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        features = self.conv(x)
        flat = features.view(x.shape[0], -1)
        latent = self.fc(flat)

        if self.use_quantum and self.quantum_layer is not None:
            latent = self.quantum_layer(latent)

        logits = self.out(latent)
        return logits


__all__ = ["FraudDetectionHybridModel"]
