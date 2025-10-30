import pennylane as qml
import numpy as np

class QuantumAttentionCircuit:
    """
    Variational circuit that produces a probability distribution over
    token pairs, interpreted as attention scores.  The circuit is
    parameterised by rotation angles and two‑qubit entangling gates.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter_shift")
        def circuit(params, entangle):
            # params shape: (n_qubits, 3) for RX, RY, RZ
            for w, (rx, ry, rz) in enumerate(params):
                qml.RX(rx, wires=w)
                qml.RY(ry, wires=w)
                qml.RZ(rz, wires=w)
            # Entanglement
            for w in range(n_qubits - 1):
                qml.CRX(entangle[w], wires=[w, w + 1])
            # Measure all qubits in computational basis
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return a flattened probability vector
        that will be reshaped into an attention matrix by the caller.
        """
        probs = self.circuit(rotation_params, entangle_params)
        return probs.detach().numpy()

__all__ = ["QuantumAttentionCircuit"]
