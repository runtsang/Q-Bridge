from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler
import numpy as np

class SamplerQNNGen310:
    """Quantum sampler network with a deeper variational circuit.
    Provides a `sample` method that returns samples according to the learned distribution.
    """

    def __init__(self,
                 num_qubits: int = 2,
                 num_layers: int = 3,
                 seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.seed = seed

        # Parameter vectors
        self.input_params = ParameterVector("input", num_qubits)
        self.weight_params = ParameterVector(
            "w", num_qubits * num_layers * 2)  # two rotations per qubit per layer

        # Build the circuit
        self.circuit = self._build_circuit()

        # Sampler primitive
        self.sampler = StatevectorSampler(seed=seed)

        # Instantiate Qiskit SamplerQNN
        self.qiskit_sampler_qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input rotations
        for i in range(self.num_qubits):
            qc.ry(self.input_params[i], i)

        # Variational layers
        for l in range(self.num_layers):
            for i in range(self.num_qubits):
                qc.ry(self.weight_params[l * self.num_qubits * 2 + i * 2], i)
                qc.ry(self.weight_params[l * self.num_qubits * 2 + i * 2 + 1], i)
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.num_qubits - 1, 0)

        return qc

    def sample(self,
               inputs: np.ndarray,
               num_shots: int = 1024) -> np.ndarray:
        """Return measurement samples for given classical inputs."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        # Flatten inputs to list of parameters
        param_dict = {p: v for p, v in zip(self.input_params, inputs[0])}
        result = self.qiskit_sampler_qnn.sample(param_dict, shots=num_shots)
        return np.array(result)

    def parameters(self) -> np.ndarray:
        """Return current weight parameters."""
        return np.array(self.weight_params.params)

    def set_parameters(self, new_params: np.ndarray) -> None:
        """Set new weight parameters."""
        for p, val in zip(self.weight_params, new_params):
            p.set_val(val)

__all__ = ["SamplerQNNGen310"]
