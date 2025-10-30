"""Hybrid quantum regression model that mirrors the classical encoder.

The model uses a Qiskit circuit to encode classical features,
then applies a shallow variational ansatz, measures Z on each qubit,
and feeds the expectation values into a classical linear head.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli
from qiskit.providers.aer import AerSimulator


def generate_superposition_data(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(Dataset):
    """Dataset that returns quantum states and regression targets."""

    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_superposition_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """Quantum regression model with a classical linear head."""

    def __init__(self, num_qubits: int, depth: int = 2, backend: str = "aer_simulator"):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend

        # Parameter vector for feature encoding (RX gates)
        self.encoding = ParameterVector("x", num_qubits)

        # Variational parameters
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build the base circuit
        self.circuit = QuantumCircuit(num_qubits)
        for i, wire in enumerate(range(num_qubits)):
            self.circuit.rx(self.encoding[i], wire)

        # Variational ansatz
        idx = 0
        for _ in range(depth):
            for wire in range(num_qubits):
                self.circuit.ry(self.weights[idx], wire)
                idx += 1
            for wire in range(num_qubits - 1):
                self.circuit.cz(wire, wire + 1)

        # Classical head
        self.head = nn.Linear(num_qubits, 1)

        # Aer simulator for expectation evaluation
        self.sim = AerSimulator(method="statevector")

    def _build_expectation_circuit(self, feature_vector: np.ndarray):
        """Return the circuit with parameters bound to a single sample."""
        bound_circuit = self.circuit.copy()
        bound_circuit = bound_circuit.bind_parameters(
            dict(zip(self.encoding, feature_vector))
        )
        return bound_circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        expect_vals = torch.zeros((bsz, self.num_qubits), dtype=torch.float32, device=state_batch.device)

        for i in range(bsz):
            sample = state_batch[i].cpu().numpy()
            circ = self._build_expectation_circuit(sample)
            result = self.sim.run(circ).result()
            state_vec = result.get_statevector(circ)
            state = Statevector(state_vec)

            for j in range(self.num_qubits):
                pauli_str = "I" * j + "Z" + "I" * (self.num_qubits - j - 1)
                exp = state.expectation_value(Pauli(pauli_str)).real
                expect_vals[i, j] = exp

        expect_vals = expect_vals.to(state_batch.device)
        return self.head(expect_vals).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
