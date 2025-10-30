"""
QuantumNATHybrid – classical implementation.

This module introduces a hybrid neural network that combines a
convolutional backbone, a small MLP (Estimator‑QNN style), and a
differentiable quantum sub‑module implemented with torchquantum.
It also provides an `evaluate` method that follows the FastBaseEstimator
interface, supporting optional Gaussian shot noise.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Turn a 1‑D list of floats into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuantumNATHybrid(nn.Module):
    """
    Classical + differentiable quantum neural network.

    The network consists of:
        * A 2‑D CNN backbone (8 → 16 feature maps).
        * A small fully‑connected MLP (Estimator‑QNN style) that maps
          the flattened CNN features to 4 outputs.
        * A torchquantum variational block that receives a 4‑bit encoder
          of the same pooled input and outputs 4 additional features.
        * The two feature streams are concatenated and fed through a
          final linear layer to produce the 4‑dimensional prediction.
    """

    class _QLayer(tq.QuantumModule):
        """Variational layer used by the hybrid model."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()

        # Classical CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Small MLP (Estimator‑QNN style)
        self.mlp = nn.Sequential(
            nn.Linear(16 * 7 * 7, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

        # Quantum sub‑module
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self._QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm_q = nn.BatchNorm1d(self.n_wires)

        # Final classifier
        self.final = nn.Linear(4 + 1, 4)
        self.norm_final = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]

        # Classical path
        cls_feat = self.features(x)
        flat = cls_feat.view(bsz, -1)
        mlp_out = self.mlp(flat).squeeze(-1).unsqueeze(-1)  # shape: (bsz, 1)

        # Quantum path
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)
        q_out = self.norm_q(q_out)

        # Concatenate and classify
        concat = torch.cat([mlp_out, q_out], dim=1)
        out = self.final(concat)
        return self.norm_final(out)

    # ------------------------------------------------------------------
    # Evaluation utilities (FastBaseEstimator style)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a batch of parameter sets.

        Parameters
        ----------
        observables: iterable of callables that map the model output to a scalar.
        parameter_sets: sequence of parameter vectors; each vector is fed as a
            single input sample to the network.
        shots: optional number of quantum shots; if provided, Gaussian
            shot noise with variance 1/shots is added to each scalar.
        seed: random seed for reproducibility of shot noise.

        Returns
        -------
        List of list of floats: outer list over parameter sets, inner list over
        observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = [
            [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            for row in results
        ]
        return noisy


__all__ = ["QuantumNATHybrid"]
