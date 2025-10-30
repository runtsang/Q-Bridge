"""Hybrid self‑attention that uses a parameterised quantum circuit and a
state‑vector estimator (EstimatorQNN style) to produce attention scores.

The quantum module mirrors the classical interface but replaces the
attention weight calculation with expectation values of a Pauli‑Y
observable.  By feeding the input features as rotation angles, the
circuit learns a quantum‑aware attention mechanism that can be
directly compared to the classical version.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridSelfAttention:
    """
    Quantum self‑attention module.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match the input dimensionality).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameterised circuit: RX rotations followed by a simple entangling layer
        self.params = [Parameter(f"θ{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        for i, p in enumerate(self.params):
            self.circuit.rx(p, i)
        # Entanglement (CX chain)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

        # EstimatorQNN that evaluates the expectation value of a Y observable
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[SparsePauliOp.from_list([("Y" * n_qubits, 1)])],
            input_params=[],
            weight_params=self.params,
            estimator=self.estimator,
        )

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Compute attention‑weighted outputs using the quantum circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input features of shape (batch, n_qubits) that will be used as rotation angles.
        shots : int, optional
            Number of shots for the backend (default 1024).

        Returns
        -------
        np.ndarray
            Attention‑weighted outputs of shape (batch, n_qubits).
        """
        batch_size = inputs.shape[0]
        outputs = []

        for i in range(batch_size):
            # Map input values to rotation parameters
            param_dict = {p: inputs[i, j] for j, p in enumerate(self.params)}
            # Evaluate expectation value of the Y observable
            result = self.estimator_qnn.evaluate(param_dict)
            # result is a list of expectation values; take the first
            attn = np.abs(result[0])  # magnitude as a simple attention weight
            outputs.append(attn * inputs[i])

        return np.array(outputs)

__all__ = ["HybridSelfAttention"]
