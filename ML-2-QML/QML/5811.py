import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    A variational quantum circuit that mimics a fully connected layer.
    Supports multiple qubits and layers, and uses the parameter‑shifting rule for gradients.
    """
    def __init__(self, n_qubits: int = 4, depth: int = 3, shots: int = 1000):
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.num_params = n_qubits * depth

        @qml.qnode(self.device, interface="numpy")
        def circuit(params):
            idx = 0
            for d in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RY(params[idx], wires=q)
                    idx += 1
                # Entanglement across the qubits
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                qml.CNOT(wires=[self.n_qubits-1, 0])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        params = np.array(thetas, dtype=np.float32)
        if params.size < self.num_params:
            pad = np.zeros(self.num_params - params.size)
            params = np.concatenate([params, pad])
        else:
            params = params[:self.num_params]
        expectation = self.circuit(params)
        return np.array([expectation])
