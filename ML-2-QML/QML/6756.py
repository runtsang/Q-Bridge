"""Hybrid quantum convolution and self‑attention module.

The quantum implementation mirrors the classical pipeline:
first a quanvolution (parameterized RX rotations + random two‑qubit
entanglement) processes the input patch, then a quantum self‑attention
circuit (parameterized single‑qubit rotations and controlled‑RX gates)
aggregates the outcome.  The module exposes a `run` method that
accepts a 2‑D NumPy array and returns a scalar attention score.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator

class HybridQuantumConvAttention:
    """
    Quantum counterpart of the hybrid convolution‑attention pipeline.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 embed_dim: int = 4,
                 conv_threshold: float = 0.5,
                 shots: int = 200):
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.conv_threshold = conv_threshold
        self.shots = shots
        self.backend = AerSimulator()

        # Build circuits
        self.conv_circuit, self.conv_params = self._build_conv_circuit()
        self.attn_circuit, self.attn_params = self._build_attn_circuit()

    def _build_conv_circuit(self):
        n = self.kernel_size ** 2
        qc = QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc, theta

    def _build_attn_circuit(self):
        n = self.embed_dim
        qc = QuantumCircuit(n)
        # Parameterized single‑qubit rotations
        rot_params = [qiskit.circuit.Parameter(f"rot{i}") for i in range(n * 3)]
        for i in range(n):
            qc.rx(rot_params[3 * i], i)
            qc.ry(rot_params[3 * i + 1], i)
            qc.rz(rot_params[3 * i + 2], i)
        # Controlled‑RX entanglement
        ent_params = [qiskit.circuit.Parameter(f"ent{i}") for i in range(n - 1)]
        for i in range(n - 1):
            qc.crx(ent_params[i], i, i + 1)
        qc.measure_all()
        return qc, rot_params + ent_params

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid quantum pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Aggregated attention score (average number of |1⟩ across qubits).
        """
        # ---------- Quanvolution ----------
        data_flat = data.reshape(1, self.kernel_size ** 2)
        param_binds = []
        for row in data_flat:
            bind = {self.conv_params[i]: np.pi if val > self.conv_threshold else 0
                    for i, val in enumerate(row)}
            param_binds.append(bind)

        job_conv = execute(self.conv_circuit,
                           self.backend,
                           shots=self.shots,
                           parameter_binds=param_binds)
        result_conv = job_conv.result().get_counts(self.conv_circuit)
        conv_score = 0
        for key, val in result_conv.items():
            conv_score += sum(int(b) for b in key) * val
        conv_score /= self.shots * self.kernel_size ** 2

        # ---------- Quantum Self‑Attention ----------
        # Bind the attention circuit parameters to the conv_score
        bind_dict = {self.attn_params[i]: conv_score for i in range(len(self.attn_params))}
        job_attn = execute(self.attn_circuit,
                           self.backend,
                           shots=self.shots,
                           parameter_binds=[bind_dict])
        result_attn = job_attn.result().get_counts(self.attn_circuit)
        attn_score = 0
        for key, val in result_attn.items():
            attn_score += sum(int(b) for b in key) * val
        attn_score /= self.shots * self.embed_dim

        return attn_score

__all__ = ["HybridQuantumConvAttention"]
