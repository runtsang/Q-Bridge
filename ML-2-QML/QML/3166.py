from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridSamplerQNN:
    """
    Quantum sampler network inspired by SamplerQNN and Quantum‑NAT.
    The circuit consists of:
        * 2 input rotation gates (RY) parameterized by `input_params`
        * An entangling CX gate
        * Two blocks of weight rotations (4 RY gates each)
        * A final entangling CX gate
    The StatevectorSampler is used to obtain the probability distribution
    over the 4 computational basis states of the 2‑qubit system.
    """
    def __init__(self) -> None:
        # Parameter vectors
        self.input_params = ParameterVector('input', length=2)
        self.weight_params = ParameterVector('weight', length=4)

        # Build the variational circuit
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        # Entangling block
        qc.cx(0, 1)
        # Weight rotations (first block)
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        # Second entangling block
        qc.cx(0, 1)
        # Final weight rotations (reuse same weights for symmetry)
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        self.circuit = qc

        # Sampler primitive
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler
        )

    def __call__(self, input_vals: list[float], weight_vals: list[float]) -> dict:
        """
        Evaluate the sampler with concrete parameter values.
        Args:
            input_vals: List of 2 floats for the input parameters.
            weight_vals: List of 4 floats for the weight parameters.
        Returns:
            Dictionary mapping basis state strings to probabilities.
        """
        return self.sampler_qnn(input_vals, weight_vals)

__all__ = ["HybridSamplerQNN"]
