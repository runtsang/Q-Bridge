"""Hybrid quantum sampler with self‑attention.

This module defines SamplerQNNGen094, a quantum neural network that
combines a parameterized sampler circuit with a quantum self‑attention
block.  The circuit first prepares the input state, applies a sequence
of Ry rotations (weights), then executes a self‑attention style entangling
layer, and finally measures all qubits.  The design extends the original
SamplerQNN by adding a quantum attention mechanism, enabling richer
expressive power while remaining fully quantum‑centric.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

class SamplerQNNGen094:
    """
    Quantum hybrid sampler with self‑attention.
    Parameters
    ----------
    n_qubits : int
        Number of qubits. Default 4.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        # Parameter vectors
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", 4 * n_qubits)  # arbitrary
        self.attn_params = ParameterVector("attn", 2 * (n_qubits - 1))
        # Build circuits
        self.circuit = self._build_circuit()
        # Sampler
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Input rotations
        for i in range(self.n_qubits):
            qc.ry(self.input_params[i], i)

        # Weight rotations
        for i in range(self.n_qubits):
            for j in range(4):
                qc.ry(self.weight_params[4 * i + j], i)

        # Self‑attention entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(self.attn_params[i], i, i + 1)
        for i in range(self.n_qubits - 1):
            qc.crx(self.attn_params[self.n_qubits - 1 + i], i + 1, i)

        # Measurement
        qc.measure(qr, cr)
        return qc

    def run(
        self,
        bind_dict: dict | None = None,
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the hybrid circuit.

        Parameters
        ----------
        bind_dict : dict, optional
            Mapping from Parameter objects to numerical values. If None,
            parameters are sampled uniformly from [0, 2π).
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts.
        """
        if bind_dict is None:
            bind_dict = {
                p: np.random.uniform(0, 2 * np.pi) for p in self.circuit.parameters
            }
        bound_qc = self.circuit.bind_parameters(bind_dict)
        job = qiskit.execute(bound_qc, self.backend, shots=shots)
        return job.result().get_counts(bound_qc)

__all__ = ["SamplerQNNGen094"]
