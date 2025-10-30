"""AdvancedSamplerQNN: a richer quantum sampler circuit.

The circuit now uses three qubits and a parameterised entangling block
that allows more expressive probability distributions.  It is wrapped
in qiskit_machine_learning's SamplerQNN so that the same interface
(`forward`, `sample`) can be used as the classical counterpart.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class AdvancedSamplerQNN:
    """Quantum sampler with three‑qubit entanglement and tunable rotations."""
    def __init__(self) -> None:
        # Parameter vectors for input (2 params) and weights (6 params)
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 6)

        # Build a 3‑qubit circuit
        qc = QuantumCircuit(3)

        # Input rotations on first two qubits
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)

        # Entangling layer
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Parameterised single‑qubit rotations
        for i, qubit in enumerate(range(3)):
            qc.ry(self.weight_params[i], qubit)

        # Additional entanglement to increase expressivity
        qc.cx(0, 2)
        qc.cx(2, 1)

        # Final rotations
        for i, qubit in enumerate(range(3)):
            qc.ry(self.weight_params[3 + i], qubit)

        # Wrap in Qiskit Machine Learning SamplerQNN
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Evaluate the circuit for given input parameters and return
        the probability distribution over the three output basis states.
        """
        probs = self.sampler_qnn.predict(inputs)
        return probs

    def sample(self, inputs: list[float], num_samples: int = 1) -> list[int]:
        """
        Draw samples from the output distribution using the underlying sampler.
        """
        probs = self.forward(inputs)
        samples = self.sampler_qnn.sample(probs, num_samples)
        return samples

__all__ = ["AdvancedSamplerQNN"]
