"""Hybrid Quanvolution classifier with classical and quantum components.

The module exposes a single class `Quanvolution__gen319` that can operate in
either classical or quantum mode.  It incorporates:

* A classical convolutional filter (QuanvolutionFilter) or a quantum
  filter (QuanvolutionFilterQuantum) that encodes 2×2 image patches.
* A classifier head built by `build_classifier_circuit` from the classical
  reference.  The head can be any depth of fully‑connected layers.
* An estimator that supports deterministic evaluation and optional shot‑noise
  addition (FastEstimator).

The interface mirrors the original `Quanvolution.py` while adding quantum
capabilities for research experimentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# --------------------------------------------------------------------------- #
# Classical filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Standard 2×2 convolutional filter from the original Quanvolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


# --------------------------------------------------------------------------- #
# Quantum filter (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuanvolutionFilterQuantum(nn.Module):
    """Quantum 2×2 patch encoder using a random two‑qubit layer."""

    def __init__(self) -> None:
        super().__init__()
        import torchquantum as tq

        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        # Flatten to 28×28 to emulate MNIST
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# Estimator utilities (classical)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate neural networks for batches of inputs and observables.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Classifier factory (classical)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum variant."""
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, list(range(num_features)), weight_sizes, observables


# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class Quanvolution__gen319(nn.Module):
    """
    Hybrid quanvolution classifier that can operate in *classical* or *quantum*
    mode.  The filter stage processes 2×2 patches, followed by a configurable
    feed‑forward classifier.

    Parameters
    ----------
    mode : {"classical", "quantum"}
        Select the filter implementation.  The rest of the network is
        classical in all modes to keep the interface simple.
    depth : int
        Depth of the classifier's feed‑forward layers.
    """

    def __init__(self, mode: str = "classical", depth: int = 2) -> None:
        super().__init__()
        if mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")
        self.mode = mode
        self.filter = QuanvolutionFilter() if mode == "classical" else QuanvolutionFilterQuantum()
        num_features = 4 * 14 * 14  # 4 filters * 14×14 patches
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    # Estimator API ---------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model with optional shot noise."""
        estimator = FastEstimator(self) if shots is not None else FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["Quanvolution__gen319"]
