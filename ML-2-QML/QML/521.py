import pennylane as qml
import numpy as np

def Autoencoder(num_qubits: int, num_layers: int = 2) -> qml.QNode:
    """
    A hybrid quantum‑classical autoencoder that learns a quantum latent representation.
    The circuit accepts a classical binary input vector of length ``num_qubits`` and
    returns a reconstructed vector of the same length via expectation values of Pauli‑Z.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Basis encoding of the input
        for i, val in enumerate(inputs):
            if val > 0.5:
                qml.PauliX(i)
        # Variational block
        qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        # Decode: expectation of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit
