"""Hybrid quantum regression model with variational circuit, quanvolution filter, and noise‑aware estimation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

# ----- data generation ----------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels."""
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

# ----- dataset -------------------------------------------------------------
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic quantum states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----- quanvolution filter ------------------------------------------------
def Conv() -> QuantumCircuit:
    """Return a quanvolution circuit that acts as a filter."""
    class QuanvCircuit:
        def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 0.5):
            self.n_qubits = kernel_size ** 2
            self.circuit = QuantumCircuit(self.n_qubits)
            self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self.circuit.rx(self.theta[i], i)
            self.circuit += random_circuit(self.n_qubits, 2)
            self.circuit.measure_all()
            self.backend = Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            """Run the circuit on a single 2‑D kernel."""
            vec = data.reshape(1, self.n_qubits)
            param_binds = []
            for dat in vec:
                bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
                param_binds.append(bind)
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)
    return QuanvCircuit()

# ----- quantum model -------------------------------------------------------
class QModel(tq.QuantumModule):
    """Hybrid quantum regression model with encoder, variational layer, and measurement head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# ----- fast estimators -----------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values for a parametrised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian noise to the deterministic estimator."""
    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]],
                 *, shots: int | None = None, seed: int | None = None) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data",
           "FastBaseEstimator", "FastEstimator", "Conv"]
