"""Hybrid quantum quanvolution and self‑attention model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute

# Quantum quanvolution filter (2×2 patches → 4‑qubit measurements)
class QuantumQuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # shape [bsz, 4*14*14]

# Quantum self‑attention block (Qiskit implementation)
class QuantumSelfAttention:
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> torch.Tensor:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring[::-1], 2)  # Qiskit LSB‑first ordering
            probs[idx] = count / shots
        return torch.as_tensor(probs, dtype=torch.float32)

# Hybrid quantum model
class HybridQuanvolutionAttention(tq.QuantumModule):
    """Quantum hybrid model combining quanvolution filter and self‑attention."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.attention = QuantumSelfAttention(n_qubits=4)
        # Linear head: concatenated quanvolution + attention output
        self.linear = nn.Linear(4 * 14 * 14 + 2 ** 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quanvolution features
        features = self.qfilter(x)  # shape [bsz, 4*14*14]
        # Prepare parameters for attention from the feature vector
        # Use first 12 values as rotation params, first 3 as entangle params per sample
        rotation_params = features.detach().cpu().numpy()[:, :12]
        entangle_params = features.detach().cpu().numpy()[:, :3]
        attended_list = []
        for r, e in zip(rotation_params, entangle_params):
            attended = self.attention.run(r, e)
            attended_list.append(attended)
        attended_tensor = torch.stack(attended_list, dim=0).to(x.device)  # shape [bsz, 16]
        # Concatenate with original features
        combined = torch.cat([features, attended_tensor], dim=1)          # shape [bsz, 800]
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAttention"]
