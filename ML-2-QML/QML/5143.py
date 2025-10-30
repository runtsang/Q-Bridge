from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.random import random_circuit

class QuantumHybridSelfAttention:
    """
    Quantum circuit that blends a convolution‑style random circuit, a self‑attention entanglement stage,
    and a parameterised feed‑forward layer. The circuit is built once and then re‑used with
    different parameter sets.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 conv_depth: int = 2,
                 attn_depth: int = 1,
                 ffn_depth: int = 2,
                 threshold: float = 127):
        self.n_qubits = n_qubits
        self.conv_depth = conv_depth
        self.attn_depth = attn_depth
        self.ffn_depth = ffn_depth
        self.threshold = threshold
        self._build_base_circuit()

    def _build_base_circuit(self):
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Convolution‑style random circuit
        conv = random_circuit(self.n_qubits, self.conv_depth, measure=False)
        qc.append(conv.to_instruction(), range(self.n_qubits))
        # Self‑attention entanglement
        for i in range(self.n_qubits - 1):
            qc.crx(0.0, i, i + 1)
        # Feed‑forward layer (parameterised rotations)
        for i in range(self.n_qubits):
            qc.rz(0.0, i)
        qc.measure_all()
        self.base_circuit = qc

    def _apply_params(self,
                      rotation_params: np.ndarray,
                      entangle_params: np.ndarray):
        circ = self.base_circuit.copy()
        # rotation_params: 3 * n_qubits
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        # entangle_params: n_qubits - 1
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        return circ

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the hybrid circuit on the given backend.

        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            The quantum backend to run the circuit on.
        rotation_params : array_like
            Rotation parameters for the self‑attention block (length 3 * n_qubits).
        entangle_params : array_like
            Entanglement parameters for the self‑attention block (length n_qubits - 1).
        shots : int, optional
            Number of shots to execute.

        Returns
        -------
        dict
            Measurement counts.
        """
        circ = self._apply_params(rotation_params, entangle_params)
        compiled = transpile(circ, backend)
        qobj = assemble(compiled, shots=shots)
        job = backend.run(qobj)
        return job.result().get_counts(circ)

def SelfAttention() -> QuantumHybridSelfAttention:
    """
    Factory function that mirrors the original API.
    """
    backend = AerSimulator()
    return QuantumHybridSelfAttention()
