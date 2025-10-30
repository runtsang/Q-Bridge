from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable, Tuple

import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf

ScalarObs = Callable[[tq.QuantumDevice], complex]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

@dataclass
class FraudLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudQuantumLayer(tq.QuantumModule):
    """Quantum layer mimicking the classical fraud‑detection block."""
    def __init__(self, params: FraudLayerParams, *, clip: bool = True) -> None:
        super().__init__()
        self.n_wires = 2
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry1 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.register_buffer("phase0", torch.tensor(_clip(params.phases[0], 5.0)))
        self.register_buffer("phase1", torch.tensor(_clip(params.phases[1], 5.0)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry1(qdev, wires=1)
        self.rz0(qdev, wires=0)
        self.crx0(qdev, wires=[0, 1])
        tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)

class QFCQuantumModule(tq.QuantumModule):
    """Simplified quantum fully‑connected module inspired by Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random = tq.RandomLayer(n_ops=25, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = torch.nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class HybridEstimator:
    """Hybrid estimator that evaluates a torchquantum module and returns observables."""
    def __init__(self, quantum_module: tq.QuantumModule) -> None:
        self.quantum_module = quantum_module

    def _apply_params(self, values: Sequence[float]) -> None:
        """Assign a sequence of floats to the module's trainable parameters."""
        for param, val in zip(self.quantum_module.parameters(), values):
            param.data.fill_(float(val))

    def evaluate(
        self,
        observables: Iterable[ScalarObs],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Run the quantum module for each parameter set and gather observables."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            self._apply_params(params)
            qdev = tq.QuantumDevice(
                n_wires=self.quantum_module.n_wires,
                bsz=1,
                device="cpu",
                record_op=True,
            )
            self.quantum_module(qdev)
            row: List[complex] = []
            for obs in observables:
                val = obs(qdev)
                row.append(val)
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObs],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add Gaussian noise to emulate a finite number of measurement shots."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(mean.real, max(1e-6, 1 / shots)),
                    rng.normal(mean.imag, max(1e-6, 1 / shots)),
                )
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridEstimator", "FraudLayerParams", "QFCQuantumModule"]
