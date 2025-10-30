"""Hybrid quantum‑classical self‑attention module.

The implementation builds on the Qiskit attention circuit from the original
seed and augments it with a Quanvolution filter implemented via TorchQuantum.
The module is callable in the same way as the classical version, enabling
side‑by‑side experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer


class QuanvolutionFilter(tq.QuantumModule):
    """
    Random two‑qubit quantum kernel applied to 2×2 image patches.
    Mirrors the implementation in the QML seed but is self‑contained for this module.
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


class HybridSelfAttentionFilter:
    """
    Quantum‑classical hybrid attention block.

    * Quantum part: a Qiskit circuit that implements a rotation‑parameterized
      self‑attention style block.
    * Classical part: the QuanvolutionFilter extracts local features.
    """

    def __init__(self, n_qubits: int = 4, backend=None) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.qfilter = QuanvolutionFilter()

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
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input image tensor (batch, 1, 28, 28).
        rotation_params, entangle_params : np.ndarray
            Parameters for the quantum attention circuit.
        shots : int, optional
            Number of shots for the backend.

        Returns
        -------
        dict
            Measurement counts from the quantum circuit.
        """
        # 1. Classical feature extraction
        with torch.no_grad():
            features = self.qfilter(inputs)  # (batch, 4*14*14)

        # 2. Quantum attention – we ignore the classical features here but
        #    the API remains identical to the seed.
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


__all__ = ["HybridSelfAttentionFilter"]
