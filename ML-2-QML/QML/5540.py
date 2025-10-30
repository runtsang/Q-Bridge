"""Quantum hybrid convolutional filter using Qiskit.

This module merges concepts from:
- Conv.py: classical stride‑2 convolution.
- Quanvolution.py: quantum kernel patch encoder.
- FCL.py: fully‑connected head.
- QuantumRegression.py: dataset and model structure.

The returned object can be used as a drop‑in replacement for the quantum filter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer
from qiskit.circuit.random import random_circuit

class _QuantumKernel:
    """Quantum kernel that encodes a 2×2 patch into a 4‑qubit circuit."""
    def __init__(self,
                 n_qubits: int = 4,
                 shots: int = 256,
                 threshold: float = 0.5,
                 backend: qiskit.providers.BaseBackend | None = None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Parameterised RX gates
        for i in range(self.n_qubits):
            qc.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
        qc.barrier()
        # Randomised entangling layer
        qc += random_circuit(self.n_qubits, 2, seed=42)
        qc.measure_all()
        return qc

    def run(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of Pauli‑Z for each qubit in the patch.

        Parameters
        ----------
        patch : np.ndarray
            Shape (batch, n_qubits) with values in [0, 1].

        Returns
        -------
        np.ndarray
            Shape (batch, n_qubits) of expectation values in [0, 1].
        """
        expectations = []
        for row in patch:
            bind = {self.circuit.parameters[i]: np.pi if val > self.threshold else 0.0
                    for i, val in enumerate(row)}
            job = qiskit.execute(self.circuit,
                                 self.backend,
                                 shots=self.shots,
                                 parameter_binds=[bind])
            result = job.result()
            counts = result.get_counts(self.circuit)
            exp = np.zeros(self.n_qubits)
            for state, cnt in counts.items():
                for i in range(self.n_qubits):
                    if state[self.n_qubits - 1 - i] == "1":
                        exp[i] += cnt
            exp /= self.shots
            expectations.append(exp)
        return np.stack(expectations, axis=0)

class _QuantumHybridConvFilter(nn.Module):
    """Hybrid convolutional filter that combines a classical backbone with a quantum kernel."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 patch_size: int = 2,
                 quantum_out: int = 4,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, bias=True)
        self.quantum = _QuantumKernel(n_qubits=out_channels, shots=256)
        self.patch_size = patch_size
        self.quantum_out = quantum_out
        self.classifier = nn.Linear(out_channels * (28 // patch_size) ** 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.backbone(x)  # (batch, out_channels, H', W')
        batch, c, h, w = conv_out.shape
        patches = conv_out.reshape(batch, c, h * w).permute(0, 2, 1)  # (batch, n_patches, c)
        patches_np = patches.detach().cpu().numpy()
        quantum_features_np = self.quantum.run(patches_np)  # (batch, n_patches, out_channels)
        quantum_features = torch.tensor(quantum_features_np, dtype=torch.float32, device=x.device)
        features = quantum_features.reshape(batch, -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

def Conv() -> _QuantumHybridConvFilter:
    """Return a quantum hybrid Conv object ready for training."""
    return _QuantumHybridConvFilter()

__all__ = ["Conv"]
