import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """
    Variational quantum sampler network. Implements a two‑qubit circuit with
    alternating Ry rotations and CNOT entanglers. Parameters are split into
    input and trainable weights. The circuit is executed on a simulator and
    returns a probability distribution over the computational basis.
    """
    def __init__(self, dev: qml.Device | None = None, seed: int | None = None):
        self.dev = dev or qml.device("default.qubit", wires=2, shots=1024)
        self.seed = seed
        self.params = np.random.uniform(0, 2*np.pi, 4, requires_grad=True)

    def circuit(self, inputs: np.ndarray, weights: np.ndarray):
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)

    @qml.qnode(device=lambda: self.dev, interface="autograd")
    def sampler(self, inputs, weights):
        self.circuit(inputs, weights)
        return qml.probs(wires=[0, 1])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        probs = self.sampler(inputs, self.params)
        return probs

    def loss(self, inputs: np.ndarray, target: np.ndarray) -> float:
        probs = self.forward(inputs)
        return -np.sum(target * np.log(probs + 1e-12))

__all__ = ["SamplerQNN"]
