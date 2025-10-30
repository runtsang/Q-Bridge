import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Variational quantum sampler implemented with PennyLane.
    Two‑qubit circuit with parameterised Ry rotations for
    input and weight angles.  The class offers a ``sample``
    method that returns the probability distribution over the
    four computational basis states, and a ``sample_measurements``
    method that draws samples from that distribution.
    """
    def __init__(self, dev: qml.Device | None = None, num_qubits: int = 2):
        if dev is None:
            dev = qml.device("default.qubit", wires=num_qubits)
        self.dev = dev
        self.num_qubits = num_qubits

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Entanglement pattern
            qml.CNOT(wires=[0, 1])
            # Weight rotations
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)
            qml.CNOT(wires=[0, 1])
            for i in range(num_qubits):
                qml.RY(weights[i + num_qubits], wires=i)
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for the given input and weight parameters.
        Returns a probability distribution over the 2^num_qubits basis states.
        """
        return self.circuit(inputs, weights)

    def sample(self,
               inputs: np.ndarray,
               weights: np.ndarray,
               n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the output distribution of the circuit.
        Returns an array of shape (n_samples, 2**num_qubits) containing
        the sampled basis‑state indices.
        """
        probs = self.forward(inputs, weights)
        return np.random.choice(len(probs), size=n_samples, p=probs)
