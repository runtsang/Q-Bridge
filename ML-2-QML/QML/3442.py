"""Quantum self‑attention using Qiskit with patch‑wise encoding."""

from __future__ import annotations

import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from typing import List, Dict

class SelfAttention:
    """
    Quantum attention block that processes each 2×2 image patch with a
    4‑qubit circuit.  Each pixel value is encoded as a Y‑rotation,
    adjacent qubits are entangled with CX gates, and all qubits are
    measured.  The measurement results are interpreted as a
    distribution of attention weights for that patch.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024, backend=None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()

    def _build_circuit(self, data: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit for a single 4‑element patch.

        Parameters
        ----------
        data : np.ndarray
            1‑D array of length 4 with pixel values.

        Returns
        -------
        QuantumCircuit
            Fully constructed circuit ready for execution.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i, val in enumerate(data):
            qc.ry(val, i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(self, inputs: torch.Tensor) -> List[Dict[str, int]]:
        """
        Execute the quantum attention block over a batch of images.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of images with shape (B, 1, 28, 28).

        Returns
        -------
        List[Dict[str, int]]
            List of measurement count dictionaries, one per patch.
        """
        results: List[Dict[str, int]] = []

        for img in inputs:
            img_np = img.squeeze().cpu().numpy()
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img_np[r:r + 2, c:c + 2].flatten()
                    qc = self._build_circuit(patch)
                    job = qiskit.execute(qc, self.backend, shots=self.shots)
                    results.append(job.result().get_counts(qc))

        return results

__all__ = ["SelfAttention"]
