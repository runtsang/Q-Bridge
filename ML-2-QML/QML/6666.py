import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class SelfAttention:
    """
    Quantum self‑attention block implemented with a variational circuit.
    The circuit encodes classical inputs via angle encoding, applies a
    rotation layer, an entangling layer, and measures Pauli‑Z expectation
    values. A parameter‑shift gradient can be computed for end‑to‑end
    differentiable training.
    """

    def __init__(self, n_qubits: int, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=range(n_qubits))

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        # Classical data encoding
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)

        # Rotation layer
        for i in range(self.n_qubits):
            qml.Rot(rotation_params[3 * i], rotation_params[3 * i + 1], rotation_params[3 * i + 2], wires=i)

        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(entangle_params[i], wires=i + 1)

        # Measurement: expectation values of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024):
        """
        Execute the variational circuit and return measurement results.
        """
        @qml.qnode(self.dev, interface="numpy", shots=shots)
        def circuit():
            return self._circuit(rotation_params, entangle_params, inputs)

        return circuit()

    def gradient(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
                 inputs: np.ndarray):
        """
        Compute the gradient of the expectation values w.r.t. the parameters
        using the parameter‑shift rule. This allows the circuit to be
        integrated into a classical optimiser.
        """
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            return self._circuit(rotation_params, entangle_params, inputs)

        return qml.gradients.param_shift(circuit)(rotation_params, entangle_params)

__all__ = ["SelfAttention"]
