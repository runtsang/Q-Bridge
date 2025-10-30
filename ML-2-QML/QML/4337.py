from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
import numpy as np

class HybridQuantumSamplerQNN:
    """
    Quantum sampler that mirrors the classical HybridSamplerQNN.
    It encodes the input vector into a parameterized circuit, applies a
    quantum kernel ansatz, and samples the resulting statevector to obtain
    a probability distribution over two outcomes.
    """
    def __init__(self, num_wires: int = 2, num_support: int = 10, gamma: float = 1.0):
        """
        Parameters
        ----------
        num_wires : int, optional
            Number of qubits in the sampler circuit, by default 2.
        num_support : int, optional
            Size of the support set (unused in the quantum sampler but kept for API parity), by default 10.
        gamma : float, optional
            Kernel width parameter (kept for API parity), by default 1.0.
        """
        # Parameter vectors for input and rotation weights
        self.inputs = ParameterVector("input", num_wires)
        self.weights = ParameterVector("weight", num_wires * 2)

        # Build the circuit
        qc = QuantumCircuit(num_wires)
        for i in range(num_wires):
            qc.ry(self.inputs[i], i)
        qc.cx(0, 1)
        # Random layer emulated by Ry rotations
        for i in range(num_wires):
            qc.ry(self.weights[i], i)
        qc.cx(0, 1)
        for i in range(num_wires):
            qc.ry(self.weights[i + num_wires], i)

        # Sampler primitive
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=sampler
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit with input `x` and return the sampled
        probability distribution over the two computational basis states.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (num_wires,). Values should be in the range [-π, π].

        Returns
        -------
        np.ndarray
            Probability distribution over the first two basis states. Shape: (2,).
        """
        # Map input to the circuit parameters
        param_bindings = {p: v for p, v in zip(self.inputs, x)}
        # Sample probabilities for the first two basis states
        probs = self.sampler_qnn.sample(param_bindings, shots=1024)
        # Convert to a numpy array of shape (2,)
        return np.array([probs.get('00', 0.0), probs.get('01', 0.0)])

__all__ = ["HybridQuantumSamplerQNN"]
