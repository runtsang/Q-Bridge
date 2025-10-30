"""Quantum hybrid network with quanvolution and variational head.

The module implements a quantum‑classical network that mirrors the
original hybrid architecture but replaces the classical convolutional
filter with a quanvolution layer and the dense head with a
variational quantum circuit.  The design is inspired by the QFCModel
from the Quantum‑NAT example and incorporates a quantum encoder,
random layer, and RX/RY rotations.  The network is fully
trainable with autograd support via torchquantum.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

import torchquantum as tq
import torchquantum.functional as tqf

# Quantum circuit used as a variational head
class QuantumCircuit(tq.QuantumModule):
    """Variational circuit that maps classical features to qubit
    expectation values.  The encoder uses a Ry‑type circuit, followed
    by a random layer and RX/RY rotations on each wire.
    """
    def __init__(self, n_wires: int, backend, shots: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.backend = backend
        self.shots = shots

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

# Quantum implementation of a 2×2 quanvolution filter
class QuanvCircuit(tq.QuantumModule):
    """Quantum implementation of a 2×2 quanvolution filter."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
        super().__init__()
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the filter on a batch of 2×2 patches.

        Parameters
        ----------
        data : np.ndarray
            Shape (batch, 2, 2).  Each element is a classical value
            that will be encoded as a π rotation if it exceeds the
            threshold, otherwise 0.
        """
        batch = data.shape[0]
        param_binds = []
        for i in range(batch):
            bind = {}
            for j, val in enumerate(data[i].flatten()):
                bind[self.theta[j]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        # Compute average number of |1> outcomes per qubit
        counts = np.array([sum(int(bit) for bit in key) * val
                           for key, val in result.items()])
        probs = counts / (self.shots * batch * self.n_qubits)
        return probs

class QuantumHybridNet(tq.QuantumModule):
    """Full quantum‑classical network that mirrors the hybrid
    architecture.  It uses a quantum variational head (QuantumCircuit)
    and a quantum quanvolution filter (QuanvCircuit) for feature
    extraction.
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor (same as the classical variant)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuit(n_wires=1, backend=backend, shots=100)

        # Measurement and linear head
        self.linear_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Quantum processing
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.quantum_circuit.n_wires,
                                bsz=bsz,
                                device=x.device)
        self.quantum_circuit.encoder(qdev, x)
        self.quantum_circuit.q_layer(qdev)
        features = self.quantum_circuit.measure(qdev)
        out = self.linear_head(features)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["QuantumCircuit", "QuanvCircuit", "QuantumHybridNet"]
