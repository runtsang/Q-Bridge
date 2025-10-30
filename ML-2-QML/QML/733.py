from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as BaseSampler
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

class SamplerQNNGen409:
    """
    A parameterised quantum sampler that extends the original two‑qubit circuit.
    It uses three layers of entanglement, a depolarising noise model, and
    provides a convenient `sample` method that returns a probability
    distribution over the computational basis.
    """
    def __init__(self,
                 input_dim: int = 2,
                 weight_dim: int = 4,
                 num_shots: int = 1024,
                 noise_prob: float = 0.01) -> None:
        # Input and weight parameters
        self.inputs = ParameterVector("input", input_dim)
        self.weights = ParameterVector("weight", weight_dim)

        # Build the circuit
        qc = QuantumCircuit(input_dim)
        # Layer 1: parameterised rotations
        for i in range(input_dim):
            qc.ry(self.inputs[i], i)
        # Entangling layer
        for i in range(input_dim - 1):
            qc.cx(i, i + 1)
        # Layer 2: more rotations
        for i in range(input_dim):
            qc.ry(self.weights[i % weight_dim], i)
        # Second entangling layer
        for i in range(input_dim - 1):
            qc.cx(i + 1, i)
        # Final rotation layer
        for i in range(input_dim):
            qc.ry(self.weights[(i + 2) % weight_dim], i)

        # Noise model
        noise_model = NoiseModel()
        error = depolarizing_error(noise_prob, 1)
        noise_model.add_all_qubit_quantum_error(error, ["ry", "cx"])

        # Sampler primitive with noise
        self.sampler = BaseSampler(
            backend=AerSimulator(noise_model=noise_model),
            shots=num_shots,
        )
        # Qiskit Machine Learning wrapper
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def sample(self, input_vals: list[float], weight_vals: list[float]) -> dict[str, float]:
        """
        Execute the circuit with the supplied parameter values and return a
        dictionary mapping basis states to their estimated probabilities.
        """
        param_dict = {str(p): v for p, v in zip(self.inputs, input_vals)}
        param_dict.update({str(p): v for p, v in zip(self.weights, weight_vals)})

        probs = self.sampler_qnn.sample(params=param_dict)
        return probs

__all__ = ["SamplerQNNGen409"]
