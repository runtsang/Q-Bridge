"""Hybrid estimator for quantum circuits using Qiskit or TorchQuantum, with shot‑based sampling and a quantum FCL."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qiskit
import torchquantum as tq


class HybridEstimator:
    """Evaluate a parameterized quantum circuit for batches of parameters and observables."""

    def __init__(self, circuit: QuantumCircuit | tq.QuantumModule) -> None:
        self.circuit = circuit
        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
        else:
            self._parameters = []

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("Binding only supported for Qiskit circuits.")
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            if isinstance(self.circuit, QuantumCircuit):
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # TorchQuantum path
                bsz = len(values)
                qdev = tq.QuantumDevice(n_wires=self.circuit.n_wires, bsz=bsz, device="cpu")
                self.circuit(qdev)
                features = self.circuit.measure(qdev)
                row = [features[i].item() for i in range(len(observables))]
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
    ) -> List[List[complex]]:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("Shot evaluation only supported for Qiskit circuits.")
        results: List[List[complex]] = []
        backend = qiskit.Aer.get_backend("qasm_simulator")
        for values in parameter_sets:
            job = qiskit.execute(
                self.circuit,
                backend,
                shots=shots,
                parameter_binds=[{param: val for param, val in zip(self._parameters, values)}],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / shots
            states = np.array(list(counts.keys()), dtype=int)
            exp = np.sum(states * probs)
            results.append([exp])
        return results

    @staticmethod
    def FCL() -> QuantumCircuit:
        """Return a simple parameterized quantum circuit mimicking the classical FCL."""
        class _QuantumFCL:
            def __init__(self, n_qubits: int = 1) -> None:
                self._circuit = QuantumCircuit(n_qubits)
                self.theta = qiskit.circuit.Parameter("theta")
                self._circuit.h(range(n_qubits))
                self._circuit.barrier()
                self._circuit.ry(self.theta, range(n_qubits))
                self._circuit.measure_all()
                self.backend = qiskit.Aer.get_backend("qasm_simulator")
                self.shots = 100

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                job = qiskit.execute(
                    self._circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[{self.theta: theta} for theta in thetas],
                )
                result = job.result().get_counts(self._circuit)
                counts = np.array(list(result.values()))
                states = np.array(list(result.keys())).astype(float)
                probs = counts / self.shots
                expectation = np.sum(states * probs)
                return np.array([expectation])

        return _QuantumFCL()

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
        return states, labels

    @staticmethod
    def RegressionDataset(samples: int, num_wires: int) -> tq.QuantumModule:
        class _RegressionDataset(tq.QuantumModule):
            def __init__(self) -> None:
                super().__init__()
                self.states, self.labels = HybridEstimator.generate_superposition_data(num_wires, samples)

            def __len__(self) -> int:  # type: ignore[override]
                return len(self.states)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
                return {
                    "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32),
                }

        return _RegressionDataset()

    @staticmethod
    def QModel(num_wires: int) -> tq.QuantumModule:
        class _QModel(tq.QuantumModule):
            def __init__(self) -> None:
                super().__init__()
                self.n_wires = num_wires
                self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)
                self.measure = tq.MeasureAll(tq.PauliZ)
                self.head = nn.Linear(num_wires, 1)

            def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
                bsz = state_batch.shape[0]
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
                self.encoder(qdev, state_batch)
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                features = self.measure(qdev)
                return self.head(features).squeeze(-1)

        return _QModel()

__all__ = ["HybridEstimator"]
