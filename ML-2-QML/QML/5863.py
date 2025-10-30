"""
SamplerQNNGen263 – Quantum sampler using a hardware‑efficient ansatz.
The circuit is a three‑layer repeat of Ry rotations and CX entanglers, with
parameter vectors for both input angles and trainable weights.  A
StatevectorSampler is used to obtain exact sampling probabilities.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import Sampler as QSampler

class SamplerQNNGen263:
    """
    Builds a parameterised quantum sampler network.
    The interface mimics the classical SamplerQNNGen263 for easy model swapping.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 3) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", n_qubits * n_layers)
        self.circuit = self._build_circuit()
        self.sampler = QSampler()
        self.qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """
        Constructs a hardware‑efficient ansatz:
          • Ry rotations for each input qubit
          • CX entanglement across all qubit pairs
          • Repeated layers of Ry rotations (weights) and CX
        """
        qc = QuantumCircuit(self.n_qubits)
        # Input layer
        for q in range(self.n_qubits):
            qc.ry(self.input_params[q], q)

        # Entangling + parameterised layers
        for layer in range(self.n_layers):
            # Apply Ry rotations with trainable weights
            for q in range(self.n_qubits):
                qc.ry(self.weight_params[layer * self.n_qubits + q], q)
            # CX entanglement (full connectivity)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            # Wrap around for odd qubits
            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)

        return qc

    def get_qnn(self) -> QSamplerQNN:
        """
        Returns the underlying Qiskit SamplerQNN instance.
        """
        return self.qnn

    def __call__(self, *args, **kwargs):
        """
        Forward pass proxy to the underlying QSamplerQNN.
        Expects a dictionary of input parameters matching self.input_params.
        """
        return self.qnn(*args, **kwargs)

__all__ = ["SamplerQNNGen263"]
