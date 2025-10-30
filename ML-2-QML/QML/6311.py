"""Hybrid fully‑connected layer with quantum implementation.

The class `HybridFCL` is a `torchquantum.QuantumModule` that encodes
classical inputs into a quantum state, applies a variational circuit,
measures expectation values, and feeds them into a classical head.
It also exposes a `run(thetas)` method that runs a simple Qiskit
parameterized circuit for quick evaluation, mirroring the original
FCL interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, execute, Aer

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form
        cos(theta)|0...0> + e^(i phi)*sin(theta)|1...1>.
    Returns the state vectors and a target label.
    """
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
    """
    Dataset that returns quantum state vectors and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridFCL(tq.QuantumModule):
    """
    Quantum hybrid fully‑connected layer.

    The module consists of:
      * a classical encoder that maps input amplitudes to a quantum state,
      * a variational layer (`QLayer`) built from random gates and
        parameterized RX/RY rotations,
      * measurement of Pauli‑Z expectation values,
      * a classical linear head that produces a scalar output.
    """
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

    def __init__(self, num_wires: int = 1):
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

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute a simple Qiskit parameterized circuit for the given thetas
        and return the expectation value of Z.
        """
        n_qubits = self.n_wires
        circuit = QuantumCircuit(n_qubits)
        circuit.h(range(n_qubits))
        for i, theta in enumerate(thetas):
            circuit.ry(theta, i)
        circuit.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend, shots=1024)
        result = job.result().get_counts(circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / 1024
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

def FCL() -> HybridFCL:
    """
    Factory function that returns an instance of the quantum hybrid layer.
    """
    return HybridFCL()
