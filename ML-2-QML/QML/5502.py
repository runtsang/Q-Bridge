"""Quantum‑kernel framework using TorchQuantum and Qiskit primitives."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable, List, Sequence, Callable
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuantumKernel(tq.QuantumModule):
    """Feature‑map based on a random circuit and a self‑attention style entangler."""

    def __init__(self, n_wires: int = 4, num_ops: int = 8, seed: int | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Random encoding of each input dimension
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Random layer to increase expressivity
        self.random_layer = tq.RandomLayer(n_ops=num_ops, wires=list(range(n_wires)))
        # Simple self‑attention style entangler (placeholder)
        self.attention = tq.CNOTGate(wires=list(range(n_wires - 1)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        self.attention(self.q_device)
        self.encoder(self.q_device, -y)
        self.random_layer(self.q_device)
        self.attention(self.q_device)
        return torch.abs(self.q_device.states.view(-1)[0])

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(ax, by).item() for by in b] for ax in a])


class FastBaseEstimator:
    """Expectation‑value evaluator for Qiskit circuits."""

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            circ = self._bind(vals)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class QuantumSelfAttention(tq.QuantumModule):
    """Self‑attention style block implemented with parameter‑ized gates."""

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_qubits)
        self.rotation = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(n_qubits)
        ]
        self.entangle = [
            {"input_idx": [i], "func": "crx", "wires": [i, i + 1]}
            for i in range(n_qubits - 1)
        ]

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.rotation:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in self.entangle:
            params = y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuanvolutionFilter(tq.QuantumModule):
    """Applies a 2×2 patch quantum kernel across a 28×28 image."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.random = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        img = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, r, c],
                        img[:, r, c + 1],
                        img[:, r + 1, c],
                        img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumKernelMethod:
    """Unified interface exposing classical RBF and quantum kernels with fast estimators."""

    def __init__(self, mode: str = "quantum", n_wires: int = 4, num_ops: int = 8) -> None:
        self.mode = mode
        if mode == "quantum":
            self.kernel = QuantumKernel(n_wires, num_ops)
        else:
            raise NotImplementedError("Classical mode requires the ML module.")

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.matrix(a, b)

    def evaluate(
        self,
        model: QuantumCircuit,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        estimator = FastBaseEstimator(model)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "QuantumKernelMethod",
    "QuantumKernel",
    "FastBaseEstimator",
    "QuantumSelfAttention",
    "QuanvolutionFilter",
]
