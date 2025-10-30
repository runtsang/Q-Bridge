import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Quantum‑enhanced self‑attention block built with Pennylane.
    The circuit encodes the input sequence as rotation angles, applies a
    parameterised ansatz, and measures expectation values that are
    interpreted as attention probabilities.
    """
    def __init__(self, n_qubits: int = 4, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_ansatz()

    def _build_ansatz(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, rotation_params, entangle_params):
            # Encode inputs as Ry rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Parameterised rotation layer
            for layer in range(self.num_layers):
                for i in range(self.n_qubits):
                    qml.RY(rotation_params[layer, i], wires=i)
                # Entangling layer using controlled‑RZ
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                # Entanglement with external params
                for i in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[layer, i], wires=[i, i + 1])

            # Measure expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self,
            inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the variational circuit and convert the expectation
        values into a probability distribution that acts as attention
        weights.  The returned array has shape (n_qubits,).
        """
        # Normalize inputs to [-π, π]
        norm_inputs = np.clip(inputs, -np.pi, np.pi)
        exp_vals = self.circuit(norm_inputs, rotation_params, entangle_params)
        # Convert to probabilities
        probs = np.exp(exp_vals) / np.sum(np.exp(exp_vals))
        return probs

__all__ = ["SelfAttention"]
