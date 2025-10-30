"""Hybrid quantum model that fuses a quanvolution filter, quantum self‑attention
and a quantum sampler.

The module is fully compatible with the original ``Quanvolution`` interface
and demonstrates how a quantum convolution can be enriched with two
additional quantum blocks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit, Aer


class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        # Convert counts to a probability vector
        probs = np.array(
            [counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2**self.n_qubits)],
            dtype=np.float32,
        )
        return probs / shots


class QuantumSamplerQNN:
    """Simple parameterised quantum sampler with two qubits."""

    def __init__(self) -> None:
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        qc = QuantumCircuit(2)
        # Encode inputs as Ry rotations
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        probs = np.array(
            [counts.get(f"{i:02b}", 0) for i in range(4)], dtype=np.float32
        )
        return probs / shots


class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum hybrid model that combines a quanvolution filter,
    a quantum self‑attention block and a quantum sampler."""

    def __init__(self) -> None:
        super().__init__()
        # Quanvolution filter
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

        # Quantum sampler
        self.sampler = QuantumSamplerQNN()

        # Linear head: 4*14*14 (filter) + 16 (attention counts) + 4 (sampler counts)
        self.linear = nn.Linear(4 * 14 * 14 + 16 + 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device

        # Quanvolution filter
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        filter_feat = torch.cat(patches, dim=1)  # (bsz, 4*14*14)

        # Quantum self‑attention
        rot_params = np.random.randn(4 * 3)
        ent_params = np.random.randn(4 - 1)
        attn_probs = self.attention.run(rot_params, ent_params, shots=1024)
        attn_tensor = torch.as_tensor(attn_probs, dtype=torch.float32, device=device).unsqueeze(0).repeat(bsz, 1)

        # Quantum sampler
        samp_inputs = np.random.rand(2)
        samp_probs = self.sampler.run(samp_inputs, shots=1024)
        samp_tensor = torch.as_tensor(samp_probs, dtype=torch.float32, device=device).unsqueeze(0).repeat(bsz, 1)

        # Concatenate all signals
        combined = torch.cat([filter_feat, attn_tensor, samp_tensor], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
