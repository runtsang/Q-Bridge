import pennylane as qml
from pennylane import numpy as np

def autoencoder_qnn(
    latent_dim: int,
    qubits: int = 8,
    depth: int = 3,
    seed: int | None = 42,
) -> qml.QNode:
    """
    Returns a Pennylane QNode that implements the quantum regulariser
    used in the hybrid autoencoder. The circuit encodes the latent
    vector into rotations on the first `min(latent_dim, qubits)` qubits,
    applies `depth` parameterised layers and returns the expectation
    value of Pauli‑Z on the first qubit.
    """
    dev = qml.device("default.qubit", wires=qubits)
    np.random.seed(seed)
    params = np.random.randn(qubits * depth * 3)

    @qml.qnode(dev, interface="jax")
    def circuit(latent: np.ndarray) -> np.ndarray:
        # Encode latent
        for i in range(min(latent_dim, qubits)):
            qml.RX(latent[i], wires=i)

        # Parameterised layers
        p_idx = 0
        for _ in range(depth):
            for w in range(qubits):
                qml.RY(params[p_idx], wires=w); p_idx += 1
                qml.RZ(params[p_idx], wires=w); p_idx += 1
                qml.CNOT(wires=[w, (w + 1) % qubits]); p_idx += 1

        return qml.expval(qml.PauliZ(0))

    return circuit
