"""Classical hybrid classifier mirroring the quantum interface.

The class implements a feed‑forward network that mimics the structure of the
quantum circuit: an encoding layer followed by a configurable depth of hidden
layers and a binary classification head.  The static method
`build_classifier_circuit` returns the model together with metadata that
parallels the quantum version, enabling seamless comparison or substitution.

The module also provides a lightweight sampler network for data augmentation
and regression utilities for auxiliary experiments.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = [
    "HybridClassifierModel",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
]


class HybridClassifierModel(nn.Module):
    """Classical hybrid classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input features.
    depth : int, default=2
        Number of hidden layers after the encoding layer.
    """

    def __init__(self, num_features: int, depth: int = 2) -> None:
        super().__init__()
        self.encoding = nn.Linear(num_features, num_features)
        hidden_layers: List[nn.Module] = [self.encoding, nn.ReLU()]
        for _ in range(depth):
            hidden_layers.append(nn.Linear(num_features, num_features))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.head = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits for two classes."""
        return self.head(self.hidden(x))

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Return a tuple mimicking the quantum build function:
        (model, encoding_indices, weight_sizes, observables)

        The encoding_indices correspond to the positions of the input
        features in the network; weight_sizes list the number of trainable
        parameters per linear layer; observables are dummy indices for
        compatibility with the quantum API.
        """
        model = HybridClassifierModel(num_features, depth)
        encoding = list(range(num_features))
        weight_sizes = [
            layer.weight.numel() + layer.bias.numel()
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
        ]
        observables = list(range(2))
        return model, encoding, weight_sizes, observables


def SamplerQNN() -> nn.Module:
    """Simple classical sampler network for data augmentation.

    The network maps a 2‑dimensional input into a probability distribution
    over two classes.  It is intentionally lightweight to keep the focus on
    the hybrid architecture.
    """
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


def generate_superposition_data(
    num_features: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data resembling a superposition of basis states.

    The output ``y`` is a noisy sine function of the sum of the input
    features, providing a simple regression target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset for the regression task.

    It returns a pair of tensors: ``states`` (features) and ``target`` (label).
    """

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }
