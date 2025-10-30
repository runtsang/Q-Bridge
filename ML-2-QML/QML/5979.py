from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler
import numpy as np

class SamplerQNN:
    """
    Quantum sampler network that extends the original seed with a
    parameter‑driven entangling block and a measurement‑based output.
    The circuit accepts two input parameters (representing a 2‑D
    feature vector) and four trainable weight parameters.

    The forward method returns a probability distribution over two
    measurement outcomes, mirroring the classical network's output
    shape.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 4):
        # Parameter vectors
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", weight_dim)

        # Build the circuit
        self.circuit = QuantumCircuit(input_dim)
        # Input encoding
        for i in range(input_dim):
            self.circuit.ry(self.input_params[i], i)
        # Entangling block
        self.circuit.cx(0, 1)
        # Variational rotation block
        for i in range(weight_dim):
            self.circuit.ry(self.weight_params[i], i % input_dim)
        # Final entanglement
        self.circuit.cx(0, 1)

        # Sampler primitive
        self.sampler = StatevectorSampler()
        # Wrap in Qiskit ML SamplerQNN
        self.qiskit_sampler = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute a probability distribution over the two measurement
        outcomes for a batch of input vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch_size, 2) where each row is a 2‑D feature vector.

        Returns
        -------
        probs : np.ndarray
            Shape (batch_size, 2) with probabilities for |0⟩ and |1⟩.
        """
        # Qiskit expects 2‑D arrays with shape (batch_size, input_dim)
        probs = self.qiskit_sampler.sample(inputs, return_counts=False)
        # The sampler returns probabilities for all basis states; we
        # collapse to a 2‑class distribution by summing over the
        # second qubit: |0⟩ (first qubit 0) vs |1⟩ (first qubit 1).
        probs_2class = np.zeros((inputs.shape[0], 2))
        for i, p in enumerate(probs):
            probs_2class[i, 0] = p.get("00", 0.0) + p.get("01", 0.0)
            probs_2class[i, 1] = p.get("10", 0.0) + p.get("11", 0.0)
        return probs_2class

__all__ = ["SamplerQNN"]
