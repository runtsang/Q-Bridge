"""Hybrid quantum estimator that evaluates expectation values of a quantum module."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
import torch.nn.functional as F
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model inspired by the Quantum‑NAT paper."""

    class QLayer(tq.QuantumModule):
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
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class FastBaseEstimator:
    """Evaluate quantum expectation values for batches of parameters and observables."""

    def __init__(self, quantum_module: tq.QuantumModule | None = None) -> None:
        self.quantum_module = quantum_module if quantum_module is not None else QFCModel()

    def _set_params(self, param_values: Sequence[float]) -> None:
        """Flatten the module parameters and overwrite with provided values."""
        flat_params = list(self.quantum_module.parameters())
        idx = 0
        for p in flat_params:
            num = p.numel()
            if idx + num > len(param_values):
                raise ValueError("Parameter set too short.")
            new = torch.tensor(param_values[idx:idx + num], dtype=p.dtype, device=p.device)
            p.data.copy_(new)
            idx += num
        if idx!= len(param_values):
            raise ValueError("Parameter set too long.")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of qiskit BaseOperator instances.
        parameter_sets:
            Iterable of parameter vectors to feed into the quantum module.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            # Set module parameters
            self._set_params(values)

            # Prepare dummy input for forward pass
            dummy = torch.zeros((1, 1, 28, 28), device="cpu")
            self.quantum_module.forward(dummy)

            # After forward, the device holds the final state; evaluate observables
            row: List[complex] = []
            for obs in observables:
                exp = self.quantum_module.measure.expectation_value(obs)
                # Take mean over batch for a single scalar value
                row.append(complex(exp.mean().item()))
            results.append(row)

        return results


__all__ = ["FastBaseEstimator", "QFCModel"]
