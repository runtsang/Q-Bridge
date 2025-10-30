"""Enhanced classical classifier builder with residual blocks and dropout.

The `ClassifierCircuitBuilder` class mirrors the interface of the seed but
adds depth‑wise residual connections, batch normalisation and optional
dropout.  It retains the `build_classifier_circuit` function for legacy
compatibility.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class ClassifierCircuitBuilder:
    """Factory for residual feed‑forward classifiers."""
    
    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a deep residual classifier with batch normalisation.

        Parameters
        ----------
        num_features:
            Dimensionality of the input feature vector.
        depth:
            Number of residual blocks.

        Returns
        -------
        model:
            nn.Sequential containing the classifier.
        encoding:
            List of feature indices used for the encoding stage
            (identical to the seed).
        weight_sizes:
            Number of learnable parameters per layer, useful for
            logging and pruning experiments.
        observables:
            Dummy list of output indices (mirrors the seed).
        """
        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(in_dim, num_features),
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(num_features, num_features),
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True),
            )
            # Residual connection
            layers.append(nn.Sequential(block, nn.Identity()))
            in_dim = num_features

        # Final classification head
        head = nn.Linear(in_dim, 2)
        layers.append(head)

        # Assemble the model
        model = nn.Sequential(*layers)

        # Compute weight sizes
        weight_sizes = [
            sum(p.numel() for p in layer.parameters()) for layer in model
        ]

        encoding = list(range(num_features))
        observables = list(range(2))
        return model, encoding, weight_sizes, observables


# Back‑compatibility shim
build_classifier_circuit = ClassifierCircuitBuilder.build_classifier_circuit

__all__ = ["ClassifierCircuitBuilder", "build_classifier_circuit"]
