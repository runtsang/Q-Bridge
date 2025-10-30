"""Hybrid quantum self‑attention circuit that combines the attention block from the original
SelfAttention Qiskit implementation with a quanvolution‑style two‑qubit kernel
implemented via torchquantum. The class exposes a run method that accepts a backend
and returns measurement counts.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import torch
import torchquantum as tq  # type: ignore


class HybridQuantumSelfAttention(tq.QuantumModule):
    """Quantum module that first applies a random two‑qubit kernel (a simplified
    quanvolution) to each 2×2 image patch and then runs a quantum attention
    circuit on the encoded qubits.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_wires = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _attention_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(qr, cr)
        return circ

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        backend=None,
        shots: int = 1024,
    ) -> dict:
        """
        Apply the quanvolution kernel to each 2×2 patch, then run the attention
        circuit on the resulting qubits and return measurement counts.
        """
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        # Encode image patches into a quantum device
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
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
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patch measurements
        qfeat = torch.cat(patches, dim=1)
        # Now run the attention circuit for each batch element
        counts = {}
        for i in range(bsz):
            circ = self._attention_circuit(rotation_params, entangle_params)
            job = execute(circ, backend, shots=shots)
            counts[i] = job.result().get_counts(circ)
        return counts


__all__ = ["HybridQuantumSelfAttention"]
