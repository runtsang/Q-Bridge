"""Quantum self‑attention implementation that mirrors the hybrid PyTorch head.

The class builds a parameterised quantum circuit whose measurement outcomes
directly encode the attention weight distribution. It can be executed on any
backend that supports sampling, and the resulting probabilities are returned
as a NumPy array.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
import numpy as np

class SelfAttentionGen093Quantum:
    """Quantum self‑attention block that outputs a probability vector over ``n_qubits``."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the parameterised circuit used for attention weight generation."""
        self.input_params = ParameterVector("input", self.n_qubits)
        self.weight_params = ParameterVector("weight", self.n_qubits * 3)

        self.circuit = QuantumCircuit(self.qr, self.cr)

        for i in range(self.n_qubits):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[3 * i], i)
            self.circuit.ry(self.weight_params[3 * i + 1], i)
            self.circuit.rz(self.weight_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        self.circuit.measure(self.qr, self.cr)

    def run(self,
            backend,
            inputs: np.ndarray,
            weight_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the attention circuit and return a probability distribution.

        Parameters
        ----------
        backend
            Qiskit backend (simulator or real device).
        inputs : np.ndarray
            Array of shape ``(n_qubits,)`` containing rotation angles for the input parameters.
        weight_params : np.ndarray
            Array of shape ``(n_qubits * 3,)`` containing rotation angles for the weight parameters.
        shots : int, optional
            Number of shots for the measurement, by default 1024.

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(n_qubits,)``.
        """
        if inputs.shape[0]!= self.n_qubits:
            raise ValueError("Input array length must match ``n_qubits``.")
        if weight_params.shape[0]!= self.n_qubits * 3:
            raise ValueError("Weight array length must be ``n_qubits * 3``.")

        bound_params = {f"input_{i}": inputs[i] for i in range(self.n_qubits)}
        bound_params.update({f"weight_{i}": weight_params[i] for i in range(self.n_qubits * 3)})

        bound_circuit = self.circuit.bind_parameters(bound_params)

        job = execute(bound_circuit, backend, shots=shots)
        counts = job.result().get_counts(bound_circuit)

        probs = np.array([counts.get(bin(i), 0) / shots for i in range(self.n_qubits)], dtype=np.float64)
        return probs

__all__ = ["SelfAttentionGen093Quantum"]
