import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

class SamplerQNNEnhanced:
    """
    Enhanced quantum sampler network with 4 qubits, deeper entanglement, and a
    state‑vector sampler backend. Provides richer probability distributions
    and flexible sampling capabilities.
    """
    def __init__(self, num_qubits: int = 4, input_dim: int = 2, weight_dim: int = 8):
        self.num_qubits = num_qubits
        self.input_params = ParameterVector('x', input_dim)
        self.weight_params = ParameterVector('w', weight_dim)

        self.circuit = QuantumCircuit(num_qubits)

        # Input encoding using Ry rotations
        for i, param in enumerate(self.input_params):
            self.circuit.ry(param, i)

        # First entangling layer
        for i in range(num_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Parameterized rotation layers
        for w in self.weight_params:
            self.circuit.ry(w, w.index % num_qubits)

        # Second entangling layer
        for i in range(num_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Sampler primitive with state‑vector backend
        self.sampler = Sampler(backend=Aer.get_backend('statevector_simulator'))

        # Wrap the circuit as a Qiskit Machine Learning SamplerQNN
        self.model = QSamplerQNN(circuit=self.circuit,
                                 input_params=self.input_params,
                                 weight_params=self.weight_params,
                                 sampler=self.sampler)

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum sampler on a batch of input vectors.
        Returns the measurement probabilities for each qubit.
        """
        return self.model.predict(inputs)

    def sample(self, inputs: np.ndarray, num_shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the quantum sampler for each input vector.
        """
        return self.model.sample(inputs, shots=num_shots)

__all__ = ["SamplerQNNEnhanced"]
