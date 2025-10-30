"""Hybrid sampler network combining classical convolution, quantum sampling, and fully connected layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports from the seed modules
from.Conv import Conv
from.FCL import FCL
from.SamplerQNN import SamplerQNN
from.FastBaseEstimator import FastEstimator


class HybridSampler(nn.Module):
    """
    A PyTorch module that stitches together:
    * Conv – a 2‑D convolutional filter (classical)
    * SamplerQNN – a quantum sampler circuit (parameterised)
    * FCL – a quantum fully‑connected layer (parameterised)
    * a final linear classifier

    The forward pass concatenates the outputs of the three sub‑modules and
    feeds them into a 2‑class softmax classifier.  The module can be
    evaluated in batch mode via the :meth:`evaluate` method, which relies
    on the lightweight FastEstimator to optionally inject shot noise.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fcl_features: int = 1,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        self.conv = Conv()
        self.sampler = SamplerQNN()
        self.fcl = FCL()
        # A tiny classical classifier that consumes the three scalar signals
        self.fc = nn.Linear(3, out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            A batch of 2‑dimensional vectors (shape ``(batch, 2)``).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over the two output classes.
        """
        # 1. Classical convolution
        conv_out = self.conv.run(inputs.detach().cpu().numpy())
        # 2. Quantum sampler (returns a softmax‑like vector)
        sampler_out = self.sampler(inputs)  # shape (batch, 2)
        # 3. Quantum fully‑connected layer
        fcl_out = self.fcl.run([float(conv_out), float(sampler_out.squeeze().item())])
        # 4. Concatenate all signals
        combined = torch.tensor(
            [conv_out, sampler_out.squeeze().item(), fcl_out],
            dtype=torch.float32,
        ).unsqueeze(0)  # shape (1, 3)
        logits = self.fc(combined)
        return F.softmax(logits, dim=-1)

    def evaluate(
        self,
        parameter_sets: list[list[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """
        Batch‑evaluate the model for a list of parameter sets.
        Uses FastEstimator to optionally add Gaussian shot noise.

        Parameters
        ----------
        parameter_sets : list[list[float]]
            Each inner list contains the flattened parameters for the
            quantum sampler and the FCL, ordered as ``[sampler_params,
            fcl_params]``.  The convolutional filter uses no parameters.
        shots : int | None, optional
            If provided, injects shot‑noise with the given number of shots.
        seed : int | None, optional
            Random seed for reproducibility of shot noise.

        Returns
        -------
        list[list[float]]
            The softmax probabilities for each parameter set.
        """
        estimator = FastEstimator(self, shots=shots, seed=seed)
        observables = [lambda x: x]
        return estimator.evaluate(observables, parameter_sets)


__all__ = ["HybridSampler"]
