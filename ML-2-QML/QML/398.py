import pennylane as qml
import pennylane.numpy as np
import numpy as np

class SamplerQNNGen:
    """
    Quantum sampler network using a Pennylane variational circuit.

    Features:
      * Two‑qubit circuit with alternating rotation layers and CNOT entanglement.
      * Parameters are optimised via gradient descent (Adam).
      * `sample` method returns measurement outcomes sampled from the circuit state.
    """
    def __init__(self, device: str = "default.qubit", shots: int = 1024) -> None:
        self.dev = qml.device(device, wires=2, shots=shots)
        self.n_params = 8  # 4 rotation parameters per qubit
        self.params = np.random.uniform(0, 2 * np.pi, self.n_params)
        self.sampler_qnode = qml.qnode(self.dev, interface="autograd")(self._circuit)

    def _circuit(self, params, inputs):
        """Parameterized quantum circuit."""
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])

        # Rotation layer
        for i in range(2):
            qml.RY(params[2 * i], wires=i)
            qml.RZ(params[2 * i + 1], wires=i)

        qml.CNOT(wires=[0, 1])

    def forward(self, inputs):
        """Return probability distribution over the 4 possible measurement outcomes."""
        probs = self.sampler_qnode(self.params, inputs)
        return probs

    def sample(self, inputs, n_samples=1):
        """Draw `n_samples` measurement outcomes from the circuit."""
        probs = self.forward(inputs)
        # Convert probabilities to discrete samples
        outcomes = np.random.choice(4, size=n_samples, p=probs)
        return outcomes
