import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

def conv_circuit(params, wires):
    """Two‑qubit convolution block used in the QCNN."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])

def pool_circuit(params, wires):
    """Two‑qubit pooling block that reduces entanglement."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])

def create_qcnn_circuit(num_qubits: int, depth: int, feature_map: qml.QNode):
    """Build a QCNN circuit with adaptive depth.

    Args:
        num_qubits: Number of qubits in the circuit.
        depth: Number of convolution‑pooling cycles.
        feature_map: A QNode that prepares the input state.
    Returns:
        A QNode representing the full QCNN circuit.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Encode inputs using the provided feature map
        feature_map(inputs)

        # Convolution‑pooling cycles
        weight_index = 0
        for d in range(depth):
            # Convolution layer
            for i in range(0, num_qubits, 2):
                conv_circuit(weights[weight_index:weight_index+3], wires=[i, i+1])
                weight_index += 3
            # Pooling layer
            for i in range(0, num_qubits, 2):
                pool_circuit(weights[weight_index:weight_index+3], wires=[i, i+1])
                weight_index += 3

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit

def QCNNModelEnhancedQNN(num_qubits: int = 8, depth: int = 2):
    """Return a hybrid QNN that can be trained with a classical optimizer."""
    # Define a simple feature map (Z‑feature map)
    def feature_map(inputs):
        qml.feature_map(inputs, wires=range(num_qubits))

    # Build the QCNN circuit
    circuit = create_qcnn_circuit(num_qubits, depth, feature_map)

    # Trainable weights
    num_weights = depth * (num_qubits // 2) * 6  # 3 params per conv + 3 per pool
    weights = pnp.random.uniform(0, 2*np.pi, size=num_weights)

    return circuit, weights

__all__ = ["create_qcnn_circuit", "QCNNModelEnhancedQNN"]
