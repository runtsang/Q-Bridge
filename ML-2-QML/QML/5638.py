"""Hybrid quantum regression with quanvolution encoder and variational circuit."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator

# Helper to build a quanv circuit
def build_quanv_circuit(n_qubits: int, threshold: float = 0.5, shots: int = 100):
    """Return a qiskit circuit that encodes a 2‑D pattern and measures all qubits."""
    circ = qiskit.QuantumCircuit(n_qubits)
    theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        circ.rx(theta[i], i)
    circ.barrier()
    circ += random_circuit(n_qubits, 2)
    circ.measure_all()
    return circ, theta

# Pre‑build a generic quanv circuit for a given number of qubits
QUANV_CIRCUITS = {}
def get_quanv_circuit(n_qubits: int):
    if n_qubits not in QUANV_CIRCUITS:
        circ, theta = build_quanv_circuit(n_qubits)
        QUANV_CIRCUITS[n_qubits] = (circ, theta)
    return QUANV_CIRCUITS[n_qubits]

# Use a fast Aer simulator
SIMULATOR = AerSimulator()

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum data using superposition of |0> and |1>."""
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
    """Dataset that stores quantum states and labels, reshaped for 2‑D encoding."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Quantum regression model that first encodes data with a quanvolution circuit
    and then applies a variational circuit plus a classical readout."""

    class QuanvEncoder(tq.QuantumModule):
        """Encode classical data via a qiskit quanvolution circuit and use the
        resulting measurement probability as a rotation angle."""
        def __init__(self, num_wires: int, threshold: float = 0.5, shots: int = 50):
            super().__init__()
            self.num_wires = num_wires
            self.threshold = threshold
            self.shots = shots
            self.circ, self.theta = get_quanv_circuit(num_wires)
            self.backend = SIMULATOR

        def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor):
            # Compute rotation angles from the quanv circuit for each sample
            batch = state_batch.shape[0]
            angles = torch.empty(batch, dtype=torch.float32, device=state_batch.device)

            # Convert to CPU numpy for qiskit execution
            state_np = state_batch.cpu().numpy()
            for i, sample in enumerate(state_np):
                # Bind parameters based on sample values
                binds = {self.theta[j]: np.pi if val > self.threshold else 0 for j, val in enumerate(sample)}
                job = self.backend.run(self.circ.bind_parameters(binds), shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                # Compute probability of measuring |1> across all qubits
                prob = sum(
                    sum(int(bit) for bit in key) * val
                    for key, val in counts.items()
                ) / (self.shots * self.num_wires)
                angles[i] = float(prob)

            # Apply a global rotation on each wire using the computed angles
            for wire in range(self.num_wires):
                rx_gate = tq.RX(has_params=True, trainable=False)
                rx_gate(qdev, wires=wire, params=angles)

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

    def __init__(self, num_wires: int, threshold: float = 0.5, shots: int = 50):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = self.QuanvEncoder(num_wires, threshold=threshold, shots=shots)
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode data using quanvolution
        self.encoder(qdev, state_batch)
        # Apply variational layer
        self.q_layer(qdev)
        # Readout
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
