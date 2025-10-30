"""Hybrid classical classifier with optional quantum-inspired sampling.

The class exposes a consistent interface across classical and quantum
back‑ends, mirroring the original seed while adding depth‑aware metadata
and a lightweight sampler.  It can be used in pure PyTorch pipelines or
in a hybrid setting where the quantum circuit is evaluated separately.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classical_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward network that mimics the structure of the
    quantum ansatz.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential model.
    encoding : List[int]
        Indices corresponding to the input feature mapping.
    weight_sizes : List[int]
        Number of trainable parameters per linear block.
    observables : List[int]
        Dummy observable indices; kept for API compatibility.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for compatibility

    return network, encoding, weight_sizes, observables


class SamplerModule(nn.Module):
    """
    Simple feed‑forward sampler that outputs class probabilities.
    Mirrors the Qiskit SamplerQNN but remains fully classical.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) representing logits or logits+bias.

        Returns
        -------
        torch.Tensor
            Softmaxed probabilities.
        """
        return F.softmax(self.module(inputs), dim=-1)


class HybridClassifier(nn.Module):
    """
    Unified classifier that can be used in classical or quantum pipelines.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Number of hidden layers in the classical part.
    use_sampler : bool, default=True
        Whether to append the sampler module to the output.
    """

    def __init__(self, num_features: int, depth: int, use_sampler: bool = True) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classical_circuit(
            num_features, depth
        )
        self.use_sampler = use_sampler
        self.sampler = SamplerModule() if use_sampler else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network and optional sampler.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., num_features).

        Returns
        -------
        torch.Tensor
            If use_sampler, tensor of shape (..., 2) with class probabilities;
            otherwise logits of shape (..., 2).
        """
        logits = self.network(x)
        if self.use_sampler:
            return self.sampler(logits)
        return logits


__all__ = ["HybridClassifier", "build_classical_circuit", "SamplerModule"]
