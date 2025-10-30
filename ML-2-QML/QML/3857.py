import numpy as np
from qiskit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import Sampler

class HybridSamplerQNN:
    """
    Quantum sampler that implements a two‑qubit variational circuit with parameterised
    Ry rotations, a CNOT entanglement, and a fixed random layer. The circuit is sampled
    via Qiskit’s StatevectorSampler, yielding a 4‑dimensional probability distribution.
    """
    def __init__(self, seed: int | None = None) -> None:
        # Parameter vectors for inputs and weights
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build the base circuit
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)

        # Fixed random layer (reproducible via seed)
        rng = np.random.default_rng(seed)
        for i in range(4):
            angle = rng.uniform(0, 2 * np.pi)
            self.circuit.ry(angle, i % 2)  # alternate qubits

        self.circuit.cx(0, 1)

        # Initialise sampler and SamplerQNN wrapper
        self.sampler = Sampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler
        )

    def sample(self, input_vec: np.ndarray, weight_vec: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the sampler circuit with the provided input and weight parameters.
        Returns a probability distribution over the four basis states (00,01,10,11).
        """
        if input_vec.shape[0]!= len(self.inputs) or weight_vec.shape[0]!= len(self.weights):
            raise ValueError("Parameter vector sizes do not match circuit definition.")
        param_dict = {name: val for name, val in zip(self.inputs, input_vec)}
        param_dict.update({name: val for name, val in zip(self.weights, weight_vec)})

        result = self.sampler_qnn.predict(parameters=param_dict, shots=shots)
        probs = result.get_probabilities()
        # Map 4 basis states to 4 probabilities
        return np.array([probs.get(bitstring, 0.0) for bitstring in ["00", "01", "10", "11"]])

__all__ = ["HybridSamplerQNN"]
