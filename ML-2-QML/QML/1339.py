"""Quantum sampler network with enhanced entanglement and parameterised layers."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import Sampler as StateSampler


class SamplerQNNV2:
    """
    Quantum sampler network mirroring the classical SamplerQNNV2.

    Features:
    - 2 qubits with a 3‑layer parameterised circuit.
    - Each layer consists of Ry rotations on both qubits followed by a CX for entanglement.
    - Separate ParameterVectors for inputs and weights.
    - Uses Qiskit’s StatevectorSampler for efficient sampling on a simulator.
    """

    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        # Parameter vectors
        self.input_params = ParameterVector("input", num_qubits)
        self.weight_params = ParameterVector("weight", num_qubits * 3)  # 3 layers of Ry

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        # Layer 1: input rotations
        for q in range(num_qubits):
            self.circuit.ry(self.input_params[q], q)
        self.circuit.cx(0, 1)
        # Layer 2: first weight rotation
        for q in range(num_qubits):
            self.circuit.ry(self.weight_params[q], q)
        self.circuit.cx(0, 1)
        # Layer 3: second weight rotation
        for q in range(num_qubits):
            self.circuit.ry(self.weight_params[num_qubits + q], q)
        self.circuit.cx(0, 1)
        # Layer 4: final weight rotation
        for q in range(num_qubits):
            self.circuit.ry(self.weight_params[2 * num_qubits + q], q)

        # Sampler primitive
        self.sampler = StateSampler()
        # Wrap into Qiskit SamplerQNN
        self.qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def forward(self, inputs: list[float] | tuple[float,...]) -> list[float]:
        """
        Execute the quantum circuit with the given inputs and return the sampled probabilities.
        """
        # Bind input parameters
        bound_circ = self.circuit.bind_parameters(
            {param: val for param, val in zip(self.input_params, inputs)}
        )
        # Sample
        result = self.sampler.run(bound_circ, shots=1024)
        probs = result.get_counts()
        # Convert to probability distribution
        total = sum(probs.values())
        return [probs.get(f"1{q}", 0) / total for q in range(self.num_qubits)]

    def __call__(self, inputs):
        return self.forward(inputs)


__all__ = ["SamplerQNNV2"]
