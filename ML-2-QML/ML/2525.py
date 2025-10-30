"""Hybrid estimator combining classical neural networks and quantum kernels.

This module implements a lightweight estimator that can evaluate both pure
PyTorch models and hybrid models that use a quantum kernel as a feature
extractor.  The interface mirrors the original FastBaseEstimator but adds
batched evaluation, optional Gaussian shot noise, and a convenient
QuantumFeatureExtractor class inspired by the quanvolution example.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Any, Optional

import numpy as np
import torch
from torch import nn
import torchquantum as tq

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuantumFeatureExtractor(nn.Module):
    """
    A lightweight quantum kernel that maps a 2‑D image patch into a feature vector.
    The implementation uses the torchquantum library and mimics the behaviour
    of the original quanvolution filter but is exposed as a normal PyTorch
    module so it can be plugged into any network.
    """
    def __init__(self, patch_size: int = 2, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        self.n_ops = n_ops
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=self.n_ops, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated measurement vector for a batch of images."""
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, x.shape[2], self.patch_size):
            for c in range(0, x.shape[3], self.patch_size):
                patch = x[:, 0, r:r+self.patch_size, c:c+self.patch_size].view(bsz, -1)
                self.encoder(qdev, patch)
                self.layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)


class HybridEstimator:
    """
    Evaluate a model that can be either a pure PyTorch network or a hybrid
    network that uses a quantum kernel as a feature extractor.

    Parameters
    ----------
    model : nn.Module | Callable[[torch.Tensor], torch.Tensor]
        If ``model`` is a nn.Module it is used directly.  If it is a callable
        it is assumed to produce a feature vector that will be fed into a
        linear head defined by the user.
    linear_head : nn.Module, optional
        Linear head to attach to a feature extractor.  Ignored if *model* is
        already a full network.
    noise : bool, default False
        Whether to add Gaussian shot noise to the deterministic predictions.
    shots : int | None, default None
        Number of shots to use when *noise* is True.  If ``None`` the noise
        variance is set to 1/shots.
    seed : int | None, default None
        Random seed for reproducibility when noise is added.
    """
    def __init__(
        self,
        model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        *,
        linear_head: nn.Module | None = None,
        noise: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.linear_head = linear_head
        self.noise = noise
        self.shots = shots
        self.seed = seed
        if self.noise and self.shots is None:
            raise ValueError("shots must be specified when noise is enabled")
        self._rng = np.random.default_rng(seed)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, nn.Module):
            outputs = self.model(inputs)
        else:
            features = self.model(inputs)
            if self.linear_head is None:
                raise ValueError("linear_head must be provided when model is a callable")
            outputs = self.linear_head(features)
        return outputs

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                if self.noise:
                    row = [float(self._rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                results.append(row)
        return results

    def evaluate_batch(self, *args, **kwargs) -> List[List[float]]:
        """Alias for evaluate – kept for backward compatibility."""
        return self.evaluate(*args, **kwargs)

__all__ = ["HybridEstimator", "QuantumFeatureExtractor"]
