import pennylane as qml
import numpy as np
from pennylane.qnn import CircuitQNN
from pennylane.optimize import Adam

def QCNNQuantum(num_qubits: int = 8, num_layers: int = 3) -> CircuitQNN:
    """
    Construct a hybrid QCNN circuit using a RealAmplitudes feature map
    followed by a stack of convolutional and pooling layers.
    The circuit is autograd‑compatible, enabling end‑to‑end training
    with gradient‑based optimisers.
    """
    dev = qml.device("default.qubit", wires=num_qubits, shots=1024)

    def feature_map(x):
        # RealAmplitudes embedding with a single repetition
        qml.templates.embeddings.RealAmplitudes(x, wires=range(num_qubits), reps=1)

    def conv_block(wires, params):
        # Convolutional pattern: CNOT‑RY‑CNOT‑RZ on alternating qubit pairs
        for i in range(0, len(wires) - 1, 2):
            qml.CNOT(wires[i], wires[i + 1])
            qml.RY(params[i], wires[i])
            qml.CNOT(wires[i], wires[i + 1])
            qml.RZ(params[i + 1], wires[i + 1])

    def pool_block(wires, params):
        # Pooling pattern: CNOT‑RZ‑CNOT on alternating qubit pairs
        for i in range(0, len(wires) - 1, 2):
            qml.CNOT(wires[i], wires[i + 1])
            qml.RZ(params[i], wires[i])
            qml.CNOT(wires[i], wires[i + 1])

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        feature_map(inputs)
        layer_params = weights
        # Convolution + pooling stack
        for l in range(num_layers):
            conv_block(range(num_qubits), layer_params[l * 6 : (l + 1) * 6])
            pool_block(range(num_qubits), layer_params[(l + 1) * 6 : (l + 2) * 6])
        # Observable: PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    # Parameter shape: 2 * num_layers * 6
    weight_shapes = {"weights": (num_layers * 12,)}
    return CircuitQNN(circuit, weight_shapes=weight_shapes)

__all__ = ["QCNNQuantum"]
