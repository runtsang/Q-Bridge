"""Quantum QCNN with variational ansatz and noise‑aware training."""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import Adam
from pennylane.qnn import CircuitQNN
from pennylane.measurement import StateFn
from pennylane import default_qubit
from pennylane import qml
from pennylane import math

def conv_block(qubits, params, offset):
    """Two‑qubit convolution unitary parameterized by params."""
    qc = qml.QubitCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[offset], qubits[0])
    qc.ry(params[offset + 1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[offset + 2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc

def pool_block(qubits, params, offset):
    """Two‑qubit pooling unitary."""
    qc = qml.QubitCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[offset], qubits[0])
    qc.ry(params[offset + 1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[offset + 2], qubits[1])
    return qc

def build_qcnn(num_qubits=8):
    """Constructs a QCNN ansatz with alternating conv/pool layers."""
    dev = default_qubit.Device(num_qubits)
    num_params = 0
    conv_params = []
    pool_params = []

    # Feature map: ZFeatureMap (depth 1)
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Encode input features
        for i, val in enumerate(inputs):
            qml.RZ(val, i)
        # Convolutional and pooling layers
        offset = 0
        for layer in range(3):
            # Conv layer
            for q in range(0, num_qubits, 2):
                conv = conv_block([q, q+1], weights, offset)
                conv.apply()
                offset += 3
            # Pool layer
            for q in range(0, num_qubits-2, 2):
                pool = pool_block([q, q+1], weights, offset)
                pool.apply()
                offset += 3
            num_qubits //= 2
        # Measurement
        return qml.expval(qml.PauliZ(0))
    return circuit

def QCNNEnhanced():
    """Factory returning a CircuitQNN with noise‑aware training."""
    circuit = build_qcnn()
    # Define the number of trainable parameters
    num_params = 3 * (8 + 4 + 2)  # conv + pool layers
    weight_params = qml.numpy.random.uniform(0, 2*np.pi, num_params)
    qnn = CircuitQNN(circuit, weight_params)
    return qnn
