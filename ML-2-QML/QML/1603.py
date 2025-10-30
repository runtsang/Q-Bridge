import pennylane as qml
import numpy as np

class SamplerQNNAdvanced:
    """Variational quantum sampler with 2 qubits and a layered Ansatz.
    
    The circuit consists of two parameterized rotation layers followed by
    a CNOT entanglement pattern.  The parameters are split into
    input‑dependent and trainable weight sets, mirroring the structure
    of the classical sampler.  Sampling is performed on the default.qubit
    simulator and returns a probability distribution over the four basis
    states.
    """
    def __init__(self, num_qubits: int = 2, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # Trainable weight parameters: shape (num_layers, num_qubits)
        self.weight_params = np.zeros((num_layers, num_qubits), dtype=np.float64)

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Input encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(weights[l, i], wires=i)
                # Entanglement pattern
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[num_qubits - 1, 0])  # wrap‑around
            return qml.probs(wires=range(num_qubits))

        self._circuit = circuit

    def sample_distribution(self, inputs: np.ndarray | list[float]) -> np.ndarray:
        """Return the probability distribution for given inputs."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.shape!= (self.num_qubits,):
            raise ValueError(f"Expected input shape ({self.num_qubits},), got {inputs.shape}")
        probs = self._circuit(inputs, self.weight_params)
        return probs

    def trainable_parameters(self) -> np.ndarray:
        """Return the flattened array of trainable weight parameters."""
        return self.weight_params.ravel()

    def set_parameters(self, params: np.ndarray) -> None:
        """Set new weight parameters from a flattened array."""
        if params.size!= self.weight_params.size:
            raise ValueError("Parameter array has incorrect size")
        self.weight_params = params.reshape(self.weight_params.shape)
