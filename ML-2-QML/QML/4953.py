"""
Hybrid model combining a quantum convolutional filter, quantum self‑attention, and fraud‑detection style dense layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

# ---------- Quantum Self‑Attention ----------
class QuantumSelfAttention:
    """
    Variational quantum circuit implementing a self‑attention operation.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = AerSimulator()

    def _build_circuit(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, rot: np.ndarray, ent: np.ndarray, shots: int = 1024) -> torch.Tensor:
        qc = self._build_circuit(rot, ent)
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert counts to a probability vector
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)  # reverse due to Qiskit ordering
            probs[idx] = cnt
        probs /= shots
        return torch.tensor(probs, dtype=torch.float32, device="cpu")


# ---------- Hybrid Quantum Convolution ----------
class HybridQuanvolutionModel(tq.QuantumModule):
    """
    Quantum‑classical hybrid model that:
    * encodes 2×2 image patches into a 4‑qubit circuit
    * applies a random variational layer
    * measures all qubits
    * feeds the resulting feature vector into a quantum self‑attention block
    * uses fraud‑detection style dense layers for classification
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Quantum convolution parameters
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

        # Quantum self‑attention
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.attn_rotation = np.random.rand(12)   # 3 parameters per qubit
        self.attn_entangle = np.random.rand(3)    # one entangling parameter per adjacent pair

        # Fraud‑detection style dense layers
        self.fc1 = nn.Linear(4 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        """
        bsz = x.shape[0]
        device = x.device

        # Quantum convolution over 28×28 image patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        features = torch.cat(patches, dim=1)  # (batch, 4*14*14)

        # Quantum self‑attention on a fixed 4‑qubit circuit
        attn_vec = self.attention.run(self.attn_rotation, self.attn_entangle)  # (2^n,)
        attn_vec = attn_vec.unsqueeze(0).repeat(bsz, 1)  # broadcast to batch

        # Concatenate quantum convolution features with attention vector
        combined = torch.cat([features, attn_vec], dim=1)

        # Fraud‑detection style dense layers
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        logits = self.out(h)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionModel"]
