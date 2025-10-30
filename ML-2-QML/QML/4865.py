"""Hybrid quantum estimator combining fast evaluation, kernel methods, and regression."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Optional, Tuple

class FastBaseEstimator:
    """Evaluate expectation values for a parametrised circuit or a TorchQuantum module."""
    def __init__(self, circuit_or_module):
        if isinstance(circuit_or_module, QuantumCircuit):
            self._circuit = circuit_or_module
            self._parameters = list(circuit_or_module.parameters)
        elif isinstance(circuit_or_module, tq.QuantumModule):
            self._module = circuit_or_module
            self._n_wires = getattr(circuit_or_module, "n_wires", None)
        else:
            raise TypeError("Unsupported type for FastBaseEstimator: must be QuantumCircuit or TorchQuantum QuantumModule")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        if hasattr(self, "_circuit"):
            return self._evaluate_qiskit(observables, parameter_sets)
        else:
            return self._evaluate_torchquantum(observables, parameter_sets)

    def _evaluate_qiskit(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _evaluate_torchquantum(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        # Simplified evaluation assuming the module returns measurement results
        results: List[List[complex]] = []
        for values in parameter_sets:
            # Build a quantum device with a single batch element
            qdev = tq.QuantumDevice(n_wires=self._n_wires, bsz=1, device="cpu")
            # Assume the module's forward takes qdev and a state batch
            state_batch = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            self._module(qdev, state_batch)
            # Retrieve expectation values of provided observables
            row = []
            for obs in observables:
                state = Statevector(qdev.states[0].cpu().detach())
                row.append(state.expectation_value(obs))
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Simulate finite‑shot statistics by adding Gaussian noise to expectation values."""
        rng = np.random.default_rng(seed)
        raw = self.evaluate(observables, parameter_sets)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [float(rng.normal(float(val), max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy


class RegressionDataset(torch.utils.data.Dataset):
    """Quantum regression dataset generating superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate_superposition_data(num_wires, samples)

    @staticmethod
    def _generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Quantum neural network used in the regression demo."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = torch.nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that encodes two input vectors."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Simple 4‑wire quantum kernel."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "FastBaseEstimator",
    "RegressionDataset",
    "QModel",
    "Kernel",
    "KernalAnsatz",
    "kernel_matrix",
]
