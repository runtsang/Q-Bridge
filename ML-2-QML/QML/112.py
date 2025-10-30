"""Quantum sampler network that extends the original design with more qubits,
entangling layers and a hybrid classical post‑processing step.

The class builds a parameterised quantum circuit that accepts two input
parameters and four trainable weights. Additional entangling layers are added
to increase expressivity. The sampler returns measurement probabilities
which can be used in a hybrid workflow.
"""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import Sampler as QiskitSampler
import numpy as np

class SamplerQNNGen122:
    """
    A hybrid quantum sampler network with an expanded circuit.
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 4, qubits: int = 3) -> None:
        """
        Parameters
        ----------
        input_dim: int
            Number of input parameters (each maps to a rotation on a qubit).
        weight_dim: int
            Number of trainable weight parameters.
        qubits: int
            Total number of qubits used in the circuit.
        """
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.qubits = qubits
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the parameterised circuit."""
        self.inputs = ParameterVector("input", self.input_dim)
        self.weights = ParameterVector("weight", self.weight_dim)

        self.circuit = QuantumCircuit(self.qubits)
        # Encode inputs on first `input_dim` qubits
        for i in range(self.input_dim):
            self.circuit.ry(self.inputs[i], i)
        # Entangling layer
        for i in range(self.qubits - 1):
            self.circuit.cx(i, i + 1)
        # Trainable rotation layers
        for i, w in enumerate(self.weights):
            self.circuit.ry(w, i % self.qubits)
        # Additional entangling to increase expressivity
        self.circuit.cx(0, 2)
        self.circuit.cx(1, 2)

        # Final measurement
        self.circuit.measure_all()

        # Sampler primitive for execution
        self.sampler = QiskitSampler(backend=Aer.get_backend("qasm_simulator"))

        # Wrap with Qiskit Machine Learning SamplerQNN for compatibility
        self.qml_sampler = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler
        )

    def sample(self, input_values: np.ndarray, weight_values: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit with given inputs and weights and return measurement
        probabilities.

        Parameters
        ----------
        input_values: np.ndarray
            Array of shape (input_dim,) with input rotation angles.
        weight_values: np.ndarray
            Array of shape (weight_dim,) with trainable rotation angles.
        shots: int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Probability distribution over measurement outcomes.
        """
        params = dict(zip(self.inputs.params, input_values))
        params.update(dict(zip(self.weights.params, weight_values)))
        # Run sampler
        result = self.sampler.run(self.circuit, params=params, shots=shots).result()
        counts = result.get_counts()
        # Convert counts to probabilities
        probs = np.zeros(2 ** self.qubits)
        total = sum(counts.values())
        for outcome, count in counts.items():
            idx = int(outcome[::-1], 2)  # reverse bit order to match qubit ordering
            probs[idx] = count / total
        return probs

__all__ = ["SamplerQNNGen122"]
