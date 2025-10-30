"""Hybrid quantum regression model that fuses a quantum convolutional filter
with a variational circuit.

The module contains:
1. Data generation identical to the original QML seed.
2. A regression dataset that returns complex state vectors.
3. A `QuanvCircuit` quantum filter (from Conv.py) that maps a 2×2 input
   to a probability of measuring |1>.
4. A variational circuit that uses the filter probability as a rotation
   angle, followed by a random layer and a read‑out head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Data generation – identical to the original QML seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

# --------------------------------------------------------------------------- #
# Dataset – identical to the original QML seed
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum convolutional filter – borrowed from Conv.py (qiskit implementation)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """
    Quantum regression model that first runs a quantum convolutional filter
    (QuanvCircuit) on each 2×2 input patch, uses the resulting probability
    as a rotation angle, and then applies a random variational layer.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires

        # Quantum convolutional filter (2×2 → 4 qubits)
        self.quanv = QuanvCircuit(
            kernel_size=2,
            backend=qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
            threshold=127,
        )

        # Variational part
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_batch: Tensor of shape (batch, 2**num_wires) representing
                         complex amplitudes of the input state.
        Returns:
            Tensor of shape (batch,) with regression outputs.
        """
        bsz = state_batch.shape[0]
        # Compute the convolutional filter probability for each sample.
        conv_probs = torch.tensor(
            [self.quanv.run(s.numpy().reshape(2, 2)) for s in state_batch],
            dtype=torch.float32,
        )

        # Create a quantum device for the whole batch.
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)

        # Apply a rotation gate on each qubit with the filter probability.
        for i in range(bsz):
            for wire in range(self.num_wires):
                tq.RX(angle=conv_probs[i])(qdev, wires=wire)

        # Variational layer and read‑out
        self.random_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
