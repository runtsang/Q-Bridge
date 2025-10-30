"""Hybrid quantum self‑attention with quanvolution filter.

The quantum implementation mirrors the classical structure:
1. A quanvolution circuit (adapted from Conv.py) maps the input data to a
   probability distribution over qubit states.
2. A variational attention circuit (adapted from SelfAttention.py) processes
   the rotated qubit states and returns a measurement histogram.
The public ``SelfAttentionModel`` class exposes the same ``run`` method as
the classical counterpart, enabling side‑by‑side experiments.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Quanvolution filter (adapted from Conv.py)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum filter that converts classical image patches into a
    probability of measuring |1⟩ across all qubits."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the quanvolution circuit on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(kernel_size, kernel_size)``.
        Returns
        -------
        np.ndarray
            Mean probability of measuring ``|1⟩`` across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return np.array([counts / (self.shots * self.n_qubits)])

# --------------------------------------------------------------------------- #
# Quantum self‑attention core (adapted from SelfAttention.py)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Variational circuit that implements a quantum self‑attention block."""
    def __init__(self, n_qubits: int, backend):
        self.n_qubits = n_qubits
        self.backend = backend
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Execute the full quantum attention pipeline."""
        # Step 1: quanvolution filter
        quanv = QuanvCircuit(
            kernel_size=int(np.sqrt(self.n_qubits)),
            backend=self.backend,
            shots=shots,
            threshold=0.5,
        )
        conv_out = quanv.run(data)

        # Step 2: embed convolution result into qubit rotations
        circuit = self._build_circuit(rotation_params, entangle_params)
        for i, val in enumerate(conv_out[0]):
            circuit.ry(val, i)

        # Step 3: execute attention circuit
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
# Public API wrapper
# --------------------------------------------------------------------------- #
class SelfAttentionModel:
    """Quantum self‑attention model that matches the classical API."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.model = QuantumSelfAttention(n_qubits=n_qubits, backend=self.backend)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(kernel_size, kernel_size)``.
        rotation_params, entangle_params : np.ndarray
            Parameter arrays for the variational circuit.
        shots : int
            Number of shots for the back‑end simulation.
        Returns
        -------
        dict
            Measurement histogram of the final circuit.
        """
        return self.model.run(inputs, rotation_params, entangle_params, shots=shots)

__all__ = ["SelfAttentionModel"]
