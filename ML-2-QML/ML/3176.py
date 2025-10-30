"""Hybrid SamplerQNNGen108: Classical + Quantum integration.

This module defines the SamplerQNNGen108 class, a torch.nn.Module that
combines a classical feed‑forward network with a quantum sampler and a
quantum‑derived fully‑connected head.  The class exposes a `set_qsampler`
method to inject a quantum backend that implements a `run` interface
returning expectation values.  The forward pass returns both the
classical softmax distribution and the quantum‑derived output, enabling
joint optimisation.

The design follows the scaling paradigm of *combination*: it merges the
classical architecture from SamplerQNN.py with the quantum sampler
from the QML version and the fully‑connected layer from FCL.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SamplerQNNGen108(nn.Module):
    """Hybrid sampler network.

    Parameters
    ----------
    n_features : int, default 2
        Number of input features.
    hidden : int, default 4
        Size of the hidden layer in the classical part.
    out_features : int, default 2
        Number of output classes for the softmax head.
    """

    def __init__(self, n_features: int = 2, hidden: int = 4, out_features: int = 2) -> None:
        super().__init__()
        # Classical feed‑forward part
        self.classical_net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )
        # Linear layer that will consume the quantum expectation values
        self.quantum_fc = nn.Linear(1, out_features)
        # Optional quantum sampler – injected via `set_qsampler`
        self.qsampler: Optional[object] = None

    def set_qsampler(self, qsampler: object) -> None:
        """Attach a quantum sampler that implements a `run` method.

        The sampler must accept a numpy array of shape (batch, n_params)
        and return a numpy array of shape (batch, 1) containing the
        expectation value of a chosen observable.
        """
        self.qsampler = qsampler

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classical softmax output and the quantum‑derived output.

        Returns
        -------
        class_out : torch.Tensor
            Softmax probabilities from the classical network.
        quantum_out : torch.Tensor
            Linear transformation of the quantum expectation value.
        """
        # Classical branch
        class_out = F.softmax(self.classical_net(x), dim=-1)

        # Quantum branch – only if a sampler has been attached
        if self.qsampler is not None:
            # Convert to numpy for the quantum sampler
            x_np = x.detach().cpu().numpy()
            # The quantum sampler expects a 2‑D array; we supply a single
            # sample per batch element and reshape accordingly.
            q_expect = self.qsampler.run(x_np)
            # q_expect shape: (batch, 1)
            q_tensor = torch.from_numpy(q_expect.astype("float32")).to(x.device)
            quantum_out = self.quantum_fc(q_tensor)
        else:
            # Fallback: zero vector of the correct shape
            quantum_out = torch.zeros_like(class_out)

        return class_out, quantum_out

    def combined_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single distribution that mixes the classical and quantum
        branches by element‑wise addition followed by a softmax.

        This can be used when a single output is required for training.
        """
        class_out, quantum_out = self.forward(x)
        mixed = class_out + quantum_out
        return F.softmax(mixed, dim=-1)

__all__ = ["SamplerQNNGen108"]
