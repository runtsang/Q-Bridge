"""Hybrid sampler network that fuses a classical sampler with a fully‑connected layer.

The module is intentionally lightweight so it can be dropped into existing
PyTorch pipelines.  It exposes two complementary interfaces:

* ``forward(inputs, thetas)`` – the standard PyTorch forward pass.  The
  ``inputs`` are processed by a small sampler network and then fed into a
  linear head.  ``thetas`` are optional and, if supplied, are used to
  modulate the linear head weights (useful for parameter‑shuffling studies).

* ``run(thetas)`` – a pure‑Python emulation of the quantum FCL circuit.
  It accepts an iterable of angles, feeds them through a tanh‑activated
  linear layer, and returns the mean expectation value.  This method
  mirrors the behaviour of the Qiskit FCL example but stays entirely in
  NumPy/PyTorch space, enabling quick unit tests without a quantum backend.

The class is deliberately small so that it can be replaced by a full
quantum circuit later without changing the surrounding code.

Example
-------
>>> net = SamplerQNN__gen018()
>>> x = torch.randn(3, 2)          # 3 samples, 2 features
>>> y = net.forward(x)             # classical forward
>>> print(y.shape)                 # (3, 1)
>>> theta = [0.1, 0.5, 0.9]
>>> print(net.run(theta))          # quantum‑style expectation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional


class SamplerQNN__gen018(nn.Module):
    """
    Hybrid sampler network.

    Parameters
    ----------
    n_features : int, default 2
        Number of input features for the sampler.
    n_hidden : int, default 4
        Hidden width of the sampler.
    """

    def __init__(self, n_features: int = 2, n_hidden: int = 4) -> None:
        super().__init__()

        # Classical sampler: two‑layer feed‑forward network with softmax output.
        self.sampler = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_features),
        )

        # Linear head that maps the sampler output to a single logit.
        self.head = nn.Linear(n_features, 1)

        # FCL‑style linear map used by the ``run`` method.
        self.fcl = nn.Linear(1, 1)

    # ------------------------------------------------------------------ #
    #  Classical forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self,
        inputs: torch.Tensor,
        thetas: Optional[Iterable[float]] = None,
    ) -> torch.Tensor:
        """
        Forward pass that optionally modulates the linear head with
        user‑supplied angles.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, n_features)``.
        thetas : Iterable[float], optional
            If provided, the values are used to scale the head weights
            before the final linear transformation.  This mimics the
            quantum weight parameters in the sampler circuit.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, 1)`` containing the logits.
        """
        # 1. Compute probability distribution from the sampler.
        probs = F.softmax(self.sampler(inputs), dim=-1)

        # 2. Optionally modulate the head weights with thetas.
        if thetas is not None:
            thetas = torch.tensor(list(thetas), dtype=probs.dtype, device=probs.device)
            # Broadcast to match batch dimension.
            thetas = thetas.view(1, -1)
            # Modulate head weights element‑wise.
            modulated_weight = self.head.weight * thetas
            logits = F.linear(probs, modulated_weight, self.head.bias)
        else:
            logits = self.head(probs)

        return logits

    # ------------------------------------------------------------------ #
    #  Quantum‑style expectation (classical emulation)
    # ------------------------------------------------------------------ #
    def run(self, thetas: Iterable[float]) -> float:
        """
        Emulate the quantum fully‑connected layer by applying a tanh‑activated
        linear map to the supplied angles and returning the mean expectation.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of angles (weights) to feed into the linear map.

        Returns
        -------
        float
            Mean expectation value.
        """
        thetas_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.fcl(thetas_tensor)).mean()
        return expectation.item()


__all__ = ["SamplerQNN__gen018"]
