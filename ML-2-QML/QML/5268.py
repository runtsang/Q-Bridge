from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List

# Quantum fully‑connected layer (parameterized circuit)
def FCL(n_qubits: int = 1) -> QuantumCircuit:
    """Return a simple parameterized quantum circuit with a single Ry gate."""
    class QuantumFCL:
        def __init__(self, n_qubits, backend, shots):
            self._circuit = QuantumCircuit(n_qubits)
            self.theta = QuantumCircuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = tq.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    simulator = tq.Aer.get_backend("qasm_simulator")
    return QuantumFCL(n_qubits, simulator, 100)

# Dataset utilities
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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Fast estimator for quantum circuits
class FastEstimator:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Sequence[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                row = [complex(rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))) for val in row]
            results.append(row)
        return results

# Core quantum model
class QuanvolutionGen152(tq.QuantumModule):
    """
    Quantum implementation of the hybrid quanvolution model.
    Uses a 2×2 patch encoder, a random layer, and a fully connected head.
    """

    def __init__(self, n_wires: int = 4, num_classes: int = 10):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)
