import pennylane as qml
import numpy as np

# Two‑qubit device
dev = qml.device("default.qubit", wires=2)


def _variational_circuit(x, weights):
    """
    Parameterised rotations followed by a fixed entanglement pattern.
    `x` is a 2‑dimensional numpy array of classical inputs.
    `weights` is a 2‑D array of shape (num_layers, 12) where each layer
    contains six rotation parameters for each qubit (Rot gate).
    """
    # Encode inputs
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # Entanglement
    qml.CNOT(wires=[0, 1])

    # Parameterised layers
    for layer in weights:
        # First qubit
        qml.Rot(layer[0], layer[1], layer[2], wires=0)
        # Second qubit
        qml.Rot(layer[3], layer[4], layer[5], wires=1)
        # Additional entanglement
        qml.CNOT(wires=[0, 1])


@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    _variational_circuit(x, weights)
    # Return a single observable: the joint Pauli‑Z expectation
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


class EstimatorQNN:
    """
    Quantum neural network that maps a 2‑dimensional classical input to a
    scalar expectation value.  The circuit contains two parameterised rotation
    layers and a fixed CNOT entanglement.  It is compatible with autograd‑based
    optimisers from PyTorch or NumPy.
    """
    def __init__(self, num_layers: int = 2):
        # Each layer has 12 parameters (6 per qubit)
        self.weights = np.random.uniform(0, 2 * np.pi, (num_layers, 12))
        self.num_layers = num_layers

    def __call__(self, x: np.ndarray) -> float:
        return circuit(x, self.weights)

    def parameters(self):
        """Return the weight array for optimisation."""
        return self.weights


__all__ = ["EstimatorQNN"]
