"""Hybrid classical classifier with optional sampler.

The class combines a deep feed‑forward network with a lightweight
parameterized sampler.  The design mirrors the original
`build_classifier_circuit` but adds a LeakyReLU and an optional
softmax sampler so that the same object can be used in either a
pure‑classical or a hybrid setting.

The API is intentionally compatible with the legacy `QuantumClassifierModel`
module: `build_classifier_circuit` returns the network, the encoding
indices (here simply the feature indices), the weight sizes and a
list of output observables (the class logits).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridClassifierSampler(nn.Module):
    """
    Classical feed‑forward network that optionally wraps a sampler.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    use_sampler : bool, default=False
        Whether to attach a lightweight sampler network to the model.
    """

    def __init__(self, num_features: int, depth: int, use_sampler: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_sampler = use_sampler

        self.network = self._make_network()
        self.weight_sizes = self._compute_weight_sizes()

        if use_sampler:
            self.sampler = self._make_sampler()
        else:
            self.sampler = None

    def _make_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.num_features))
            layers.append(nn.LeakyReLU())
            in_dim = self.num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self) -> List[int]:
        sizes: List[int] = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    def _make_sampler(self) -> nn.Module:
        """
        A tiny neural sampler that maps logits to probabilities.
        """
        return nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        return self.network(x)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the attached sampler on the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Sample probabilities of shape (batch, 2).

        Raises
        ------
        RuntimeError
            If the sampler is not enabled.
        """
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled for this instance.")
        return self.sampler(x)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a feed‑forward network and metadata compatible with the
        legacy quantum helper interface.

        Returns
        -------
        network : nn.Module
            The constructed classifier.
        encoding : Iterable[int]
            Feature indices used as a placeholder for quantum encoding.
        weight_sizes : Iterable[int]
            Number of trainable parameters per linear layer.
        observables : List[int]
            Dummy observable indices; the classical model has no observables.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.LeakyReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = [0, 1]  # placeholder for logits
        return network, encoding, weight_sizes, observables


__all__ = ["HybridClassifierSampler"]
