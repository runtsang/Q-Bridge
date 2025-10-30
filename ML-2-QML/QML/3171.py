import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridSamplerConvQNN:
    """
    Quantum neural network that merges a parameterized sampler with a quantum
    convolution filter. The sampler part uses a 2‑parameter input and 4‑parameter
    weight vector to produce a probability distribution. The convolution part
    applies a random circuit over all qubits, encoding a 2×2 image patch into
    rotations. The output is a weighted combination of the sampler distribution
    and the average probability of measuring |1> across the qubits.
    """
    def __init__(self, kernel_size: int = 2, threshold: int = 127, shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Sampler parameters
        self.inputs2 = ParameterVector("input", 2)
        self.weights2 = ParameterVector("weight", 4)

        # Build the composite circuit
        self.circuit = QuantumCircuit(self.n_qubits)

        # Sampler part
        self.circuit.ry(self.inputs2[0], 0)
        self.circuit.ry(self.inputs2[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights2[0], 0)
        self.circuit.ry(self.weights2[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights2[2], 0)
        self.circuit.ry(self.weights2[3], 1)

        # Convolution filter part: random circuit over all qubits
        self.circuit += random_circuit(self.n_qubits, 2)

        # Sampler instance
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs2,
            weight_params=self.weights2,
            sampler=self.sampler
        )

    def run(self, data: np.ndarray) -> float:
        """
        Execute the hybrid circuit on a 2×2 image patch.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) containing pixel values.

        Returns
        -------
        float
            Combined scalar output from the quantum sampler and convolution filter.
        """
        # Encode image pixels into θ parameters (π if pixel > threshold)
        theta_bind = {
            f"theta{i}": np.pi if val > self.threshold else 0
            for i, val in enumerate(data.flatten())
        }

        # Bind sampler inputs and weights (sample values are arbitrary but fixed)
        sampler_bind = {
            self.inputs2[0]: float(data[0, 0]),
            self.inputs2[1]: float(data[0, 1]),
            self.weights2[0]: 0.1,
            self.weights2[1]: 0.2,
            self.weights2[2]: 0.3,
            self.weights2[3]: 0.4,
        }

        # Merge bindings
        bind_dict = {**theta_bind, **sampler_bind}

        # Run sampler QNN to obtain probability distribution over two outputs
        sampler_probs = self.sampler_qnn.get_probabilities(bind_dict)

        # Obtain the statevector for the full circuit
        state = self.sampler.run(self.circuit, parameter_binds=[bind_dict]).result().statevector

        # Compute probability of measuring |1> on each qubit
        probs = np.abs(state) ** 2
        bitstrings = np.array([list(format(i, f'0{self.n_qubits}b')) for i in range(2 ** self.n_qubits)], dtype=int)
        ones_per_qubit = bitstrings.T @ probs
        avg_prob = ones_per_qubit.mean()

        # Combine the outputs
        return avg_prob * sampler_probs.sum()

__all__ = ["HybridSamplerConvQNN"]
