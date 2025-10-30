"""Hybrid quantum self‑attention with a variational fully‑connected layer."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from typing import Iterable

class HybridSelfAttention:
    """
    Quantum implementation that mirrors the classical API.
    Builds a self‑attention circuit followed by a single‑qubit Ry parameter
    that emulates a fully‑connected layer.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray, theta: float) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Rotation block (3 parameters per qubit)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], qr[i])
            circuit.ry(rotation_params[3 * i + 1], qr[i])
            circuit.rz(rotation_params[3 * i + 2], qr[i])

        # Entanglement block (CRX between adjacent qubits)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], qr[i], qr[i + 1])

        # Variational fully‑connected layer (single Ry on qubit 0)
        circuit.ry(theta, qr[0])

        circuit.measure_all()
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            thetas: Iterable[float], shots: int = None) -> float:
        """
        Execute the hybrid circuit and return an expectation value.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation parameters for the self‑attention block.
        entangle_params : np.ndarray
            Entanglement parameters for the self‑attention block.
        thetas : Iterable[float]
            Parameters that modulate the variational Ry (first element used).
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        float
            Expectation value of the first qubit in the computational basis.
        """
        theta = float(next(iter(thetas), 0.0))
        circuit = self._build_circuit(rotation_params, entangle_params, theta)
        job = execute(circuit, self.backend, shots=shots or self.shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Compute expectation of Pauli‑Z on qubit 0
        exp = 0.0
        for state, count in counts.items():
            if state[0] == "1":
                exp += count
        exp /= shots or self.shots
        return exp

__all__ = ["HybridSelfAttention"]
