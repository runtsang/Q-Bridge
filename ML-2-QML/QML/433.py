import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumAttentionMask:
    """
    Parameterised variational circuit that produces a probability
    distribution over a sequence of length `seq_len`.  The circuit
    consists of a chain of single‑qubit rotations followed by
    entangling CNOT layers.  The final measurement yields a bitstring
    whose Hamming weight is used as a soft mask.

    The callable interface matches the `quantum_mask_fn` expected by
    the hybrid `SelfAttention__gen062` class.
    """
    def __init__(self, seq_len: int, n_layers: int = 2, shots: int = 8192):
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=seq_len)

        # Create a parameterised circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # params shape: (seq_len, 3) for RX, RY, RZ rotations
            for w in range(seq_len):
                qml.RX(params[w, 0], wires=w)
                qml.RY(params[w, 1], wires=w)
                qml.RZ(params[w, 2], wires=w)

            # Entangling layer
            for l in range(self.n_layers):
                for w in range(seq_len - 1):
                    qml.CNOT(wires=[w, w + 1])

            # Measure expectation value of each qubit in Z basis
            return [qml.expval(qml.PauliZ(w)) for w in range(seq_len)]

        self.circuit = circuit

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        params : np.ndarray
            Parameter array of shape (seq_len, 3) containing rotation angles.

        Returns
        -------
        np.ndarray
            Soft mask of shape (seq_len,) with values in [0, 1].
        """
        # Run the circuit and obtain expectation values
        expvals = self.circuit(params)  # list of length seq_len

        # Convert expectation values from [-1, 1] to [0, 1]
        mask = 0.5 * (np.array(expvals) + 1.0)
        return mask
