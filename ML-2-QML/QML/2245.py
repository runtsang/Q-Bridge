"""Quantum‑enhanced quanvolution with quantum self‑attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute


class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum convolutional filter that processes 2×2 patches of a 28×28 image.
    Uses a random two‑qubit layer followed by a measurement in the Z basis.
    """
    def __init__(self) -> None:
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
        return torch.cat(patches, dim=1)


class QuantumSelfAttention:
    """
    Quantum self‑attention block built with Qiskit.
    Produces an expectation‑value vector of length `n_qubits`.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
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
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert measurement counts to Z‑expectation values
        exp_vals = []
        for qubit in range(self.n_qubits):
            exp = 0.0
            for bitstring, cnt in counts.items():
                # Qiskit returns bitstrings with the most significant bit first
                bit = int(bitstring[self.n_qubits - 1 - qubit])
                exp += cnt * (1 if bit == 0 else -1)
            exp /= shots
            exp_vals.append(exp)
        return torch.tensor(exp_vals, dtype=torch.float32)


class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end quantum model that applies a quantum quanvolution filter,
    followed by a quantum self‑attention block, and a linear classifier.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.linear = nn.Linear(4 * 14 * 14 + 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape: (batch, 784)

        # Random parameters for the attention circuit
        rotation = np.random.randn(12)  # 4 qubits × 3 rotations
        entangle = np.random.randn(3)   # 3 entangling gates

        attn_out = self.attention.run(rotation, entangle)  # shape: (4,)
        # Broadcast to the batch dimension
        attn_out = attn_out.unsqueeze(0).repeat(x.size(0), 1)

        combined = torch.cat([features, attn_out], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuantumSelfAttention", "QuanvolutionClassifier"]
