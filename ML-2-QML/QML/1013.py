from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class SamplerQNNEnhanced:
    """
    Variational sampler circuit with 3 qubits and two entanglement layers.
    The circuit is parameterised by input angles and trainable weights,
    enabling richer probability distributions compared to the 2‑qubit seed.
    """

    def __init__(self, num_qubits: int = 3, entanglement: str = "full") -> None:
        self.num_qubits = num_qubits
        self.entanglement = entanglement
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.inputs = ParameterVector("input", self.num_qubits)
        self.weights = ParameterVector("weight", 2 * self.num_qubits)
        qc = QuantumCircuit(self.num_qubits)

        # Input rotations
        for i in range(self.num_qubits):
            qc.ry(self.inputs[i], i)

        # Entanglement layer 1
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        if self.entanglement == "full":
            qc.cx(self.num_qubits - 1, 0)

        # Trainable rotations
        for i in range(self.num_qubits):
            qc.ry(self.weights[i], i)

        # Entanglement layer 2
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        if self.entanglement == "full":
            qc.cx(self.num_qubits - 1, 0)

        # Final trainable rotations
        for i in range(self.num_qubits):
            qc.ry(self.weights[self.num_qubits + i], i)

        self.circuit = qc

        # Wrap in Qiskit Machine Learning SamplerQNN
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler
        )

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def sample(self, input_values: list[float], weight_values: list[float], shots: int = 1024) -> dict:
        """Execute the sampler and return measured probabilities."""
        param_bindings = {param: val for param, val in zip(self.inputs, input_values)}
        param_bindings.update({param: val for param, val in zip(self.weights, weight_values)})
        result = self.sampler_qnn.sample(param_bindings, shots=shots)
        return result
