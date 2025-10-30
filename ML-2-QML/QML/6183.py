import pennylane as qml
import numpy as np
from collections import Counter

class SelfAttentionEnhanced:
    """Quantum self‑attention using a variational circuit.

    The circuit encodes query, key and value information into qubits and
    performs a parameterised entangling layer that mimics the attention
    weighting mechanism. The return value is a probability distribution
    over the measurement outcomes, which can be interpreted as attention
    scores for a toy example.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _attention_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Encode inputs into rotation angles
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangling layer (mimicking key‑value interaction)
            for l in range(self.n_layers):
                for i in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[l * (self.n_qubits - 1) + i], wires=[i, i + 1])

            # Measure in the computational basis
            return [qml.sample(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024):
        """
        Execute the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for RX/RY/RZ gates, length 3*n_qubits.
        entangle_params : np.ndarray
            Entanglement angles, length n_layers*(n_qubits-1).
        inputs : np.ndarray
            Input data; kept for API compatibility but not used directly.
        shots : int
            Number of shots for sampling.

        Returns
        -------
        dict
            Measurement counts interpreted as attention weights.
        """
        circuit = self._attention_circuit(rotation_params, entangle_params)
        samples = circuit()
        # Convert PauliZ outcomes (-1,1) to bits (0,1)
        bitstrings = ["".join(str(int((b + 1) // 2)) for b in sample) for sample in samples]
        counts = Counter(bitstrings)
        return dict(counts)
__all__ = ["SelfAttentionEnhanced"]
