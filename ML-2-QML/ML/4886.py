"""HybridBinaryClassifier – Classical PyTorch implementation.

This module merges the classical CNN backbone from
`ClassicalQuantumBinaryClassification.py` with a compact
feed‑forward head that mimics the quantum ansatz used in the
QML counterpart.  The head is built via `build_classifier_circuit`,
so the number and depth of layers can be tuned independently.
The class exposes a `use_quantum` flag; when set to `True`
the constructor raises an error to remind the user to import
the QML variant instead.

The design follows the “FraudDetection” pattern by optionally
clipping weights and biases in the first layer, thereby
providing a safe initialisation that mirrors the photonic
model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

__all__ = ["HybridBinaryClassifier"]


def _clip(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    return torch.clamp(tensor, -bound, bound)


class _FraudLikeLinear(nn.Module):
    """Linear layer with optional clipping and a Tanh activation,
    inspired by the photonic fraud‑detection example.
    """

    def __init__(self, in_features: int, out_features: int, clip: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.clip = clip
        if clip:
            self.register_buffer("clip_min", torch.tensor(-5.0))
            self.register_buffer("clip_max", torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        b = self.linear.bias
        if self.clip:
            w = _clip(w, 5.0)
            b = _clip(b, 5.0)
        out = F.linear(x, w, b)
        return self.activation(out)


def build_classifier_circuit(num_features: int, depth: int) -> nn.Sequential:
    """Construct a purely classical dense classifier that has the same
    topology as the quantum ansatz used in the QML module.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    for i in range(depth):
        clip = i == 0  # only clip the first layer to mirror the photonic code
        layers.append(_FraudLikeLinear(in_dim, num_features, clip=clip))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


class HybridBinaryClassifier(nn.Module):
    """A CNN head followed by a dense classifier that can be swapped
    with a quantum expectation head in the QML implementation.
    """

    def __init__(
        self,
        num_features: int = 32,
        depth: int = 4,
        use_quantum: bool = False,
        shift: float = 0.0,
    ) -> None:
        super().__init__()

        if use_quantum:
            raise NotImplementedError(
                "use_quantum=True is only supported in the QML implementation."
            )

        # Simple convolutional backbone (copy of QCNet's first layers)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Compute flattened feature size after convs
        dummy_input = torch.zeros(1, 3, 32, 32)
        dummy_out = self._forward_conv(dummy_input)
        flat_size = dummy_out.shape[1]

        # Dense block that reduces dimensionality before the classifier
        self.fc_reduce = nn.Linear(flat_size, num_features)
        self.bn_reduce = nn.BatchNorm1d(num_features)

        # Classical classifier head mirroring the quantum ansatz
        self.classifier = build_classifier_circuit(num_features, depth)

        self.shift = shift

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = self.bn_reduce(self.fc_reduce(x))
        x = self.classifier(x)
        # Apply a sigmoid shift for consistency with the quantum head
        probs = torch.sigmoid(x + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)
