"""Quantum self‑attention using PennyLane.

The class implements a parameterized variational circuit that mimics a
self‑attention block.  Rotation parameters control single‑qubit rotations
while entangle parameters dictate two‑qubit CNOT‑like entanglement.  The
output is a probability distribution over measurement outcomes, which
can be interpreted as attention weights.  The interface mirrors the
original seed for easy substitution.

"""

import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational quantum self‑attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    num_layers : int, optional
        Number of alternating rotation‑entanglement layers.
    """
    def __init__(self, n_qubits: int, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray):
        """Construct a PennyLane QNode."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(*params):
            # params layout: [rotations..., entangles...]
            idx = 0
            # Apply rotation layers
            for _ in range(self.num_layers):
                for q in range(self.n_qubits):
                    qml.RX(rotation_params[idx], wires=q); idx += 1
                    qml.RY(rotation_params[idx], wires=q); idx += 1
                    qml.RZ(rotation_params[idx], wires=q); idx += 1
                # Entanglement layer (fixed CNOT pattern)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                idx = 0
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the variational circuit and return expectation values.

        Parameters
        ----------
        backend : qml.Device or None
            Pennylane device to execute on.  If None the preset device
            created in ``__init__`` is used.
        rotation_params : np.ndarray
            Array of shape ``(num_layers * n_qubits * 3,)`` containing the
            RX, RY, RZ angles for each qubit and layer.
        entangle_params : np.ndarray
            Array of shape ``(n_qubits - 1,)`` used only for API
            compatibility; the current implementation uses a fixed CNOT
            pattern.
        shots : int, optional
            Number of shots for sampling; ignored for the default autograd
            interface but kept for consistency with the seed.

        Returns
        -------
        counts : dict
            Mapping from bitstrings to estimated probabilities.
        """
        if backend is None:
            backend = self.dev
        circuit = self._build_circuit(rotation_params, entangle_params)
        # For a full probability distribution we sample
        samples = qml.sample(circuit, shots=shots)
        # Convert samples to bitstring counts
        counts = {}
        for s in samples:
            bitstring = ''.join(str(bit) for bit in s)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
