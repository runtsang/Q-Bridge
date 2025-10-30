"""Hybrid estimator for torchquantum modules or parametrized circuits.

The interface mirrors the classical estimator: evaluate a list of
observable operators for a set of parameter values.  Optional shot
noise is simulated by adding Gaussian noise to expectation values.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence as Seq

import numpy as np
import torch
from torchquantum import QuantumModule, QuantumDevice, MeasureAll, PauliZ, RandomLayer, RX, RY, RZ, CRX, hadamard, sx, cnot
from torchquantum.functional import hadamard as tq_hadamard, sx as tq_sx, cnot as tq_cnot, static as tq_static

# ----------------------------------------------------------------------
# Quantum hybrid model inspired by Quantum‑NAT
# ----------------------------------------------------------------------
class QFCModelQuantum(QuantumModule):
    """Quantum fully‑connected layer on top of a classical encoder."""

    class QLayer(QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = RX(has_params=True, trainable=True)
            self.ry0 = RY(has_params=True, trainable=True)
            self.rz0 = RZ(has_params=True, trainable=True)
            self.crx0 = CRX(has_params=True, trainable=True)

        @QuantumModule.static_support
        def forward(self, qdev: QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tq_hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tq_sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tq_cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # 4x4 RyZXY encoder (pre‑trained or random)
        self.encoder = QuantumModule.GeneralEncoder(
            QuantumModule.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = MeasureAll(PauliZ)
        self.norm = torch.nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Global pooling of the input image
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


# ----------------------------------------------------------------------
# Hybrid quantum estimator
# ----------------------------------------------------------------------
class HybridBaseEstimator:
    """Evaluate a :class:`torchquantum.QuantumModule` for batches of inputs.

    The estimator can also compute expectation values of arbitrary
    :class:`torchquantum.operators.base_operator.BaseOperator` observables.
    """

    def __init__(self, quantum_module: QuantumModule) -> None:
        self.quantum_module = quantum_module
        self._parameters = list(quantum_module.parameters())

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Seq[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            # Bind parameters to the module
            with torch.no_grad():
                for param, val in zip(self._parameters, values):
                    param.data.copy_(torch.as_tensor(val))
            # Forward pass
            out = self.quantum_module.forward(torch.empty(0))  # dummy input
            row = [obs(out) if callable(obs) else obs for obs in observables]
            results.append(row)
        return results


class HybridEstimator(HybridBaseEstimator):
    """Same as :class:`HybridBaseEstimator` but introduces shot‑noise emulation."""

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Seq[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                val + complex(rng.normal(0, max(1e-6, 1 / shots)), 0)
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridBaseEstimator", "HybridEstimator", "QFCModelQuantum"]
