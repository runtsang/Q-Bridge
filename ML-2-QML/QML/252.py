import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

class SelfAttentionEnhanced:
    """
    Variational quantum self‑attention block implemented with Pennylane.
    The circuit encodes the input sequence into qubit states, applies parameterised
    rotation and entanglement layers, and measures expectation values that are
    interpreted as attention scores.
    """

    def __init__(self, n_qubits: int = 8, num_layers: int = 2, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.device = qml.device(device_name, wires=n_qubits)

    def _encode_inputs(self, inputs: np.ndarray):
        """
        Encode a 1‑D input vector into the qubits using rotation gates.
        """
        for i, val in enumerate(inputs):
            if i < self.n_qubits:
                qml.RY(val, wires=i)

    def _ansatz(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        """
        Parameterised ansatz: rotation layers followed by entanglement layers.
        rotation_params: shape (num_layers, n_qubits, 3) for rx, ry, rz
        entangle_params: shape (num_layers, n_qubits-1) for CRX angles between neighbouring qubits
        """
        self._encode_inputs(inputs)

        for layer in range(self.num_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[layer, i, 0], wires=i)
                qml.RY(rotation_params[layer, i, 1], wires=i)
                qml.RZ(rotation_params[layer, i, 2], wires=i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[layer, i], wires=[i, i + 1])

        # Measure expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the variational circuit and return the measured expectation values.
        """
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit():
            return self._ansatz(rotation_params, entangle_params, inputs)

        result = circuit()
        # Convert from Pennylane's numpy to a plain numpy array
        return np.array(result)
