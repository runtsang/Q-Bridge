import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SamplerQNNGen212:
    """Quantum sampler network.

    Features:
    - Parameterized 2‑qubit circuit with entanglement.
    - QNode returning probability distribution over 4 basis states.
    - sample method to draw samples from the distribution.
    - trainable weights via Pennylane's gradient.
    """
    def __init__(self, dev: qml.Device | None = None):
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.params = pnp.random.uniform(0, 2*np.pi, 6)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, inputs):
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(params[2], wires=0)
            qml.RY(params[3], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(params[4], wires=0)
            qml.RY(params[5], wires=1)
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(self.params, inputs)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        probs = self.forward(inputs)
        return np.random.choice(4, size=num_samples, p=probs)

    def set_params(self, params: np.ndarray):
        self.params = params

    def get_params(self) -> np.ndarray:
        return self.params

def SamplerQNN() -> SamplerQNNGen212:
    """Factory returning an instance of SamplerQNNGen212."""
    return SamplerQNNGen212()

__all__ = ["SamplerQNN"]
