"""Quantum self‑attention layer built with Pennylane.

The implementation mirrors the classical interface: ``run`` accepts
``rotation_params`` and ``entangle_params`` (interpreted as the parameters of a
strongly entangling variational layer) together with a classical input
embedding.  The returned values are interpreted as attention weights and are
post‑processed into a probability distribution.
"""

import numpy as np
import pennylane as qml

class SelfAttention:
    """Variational quantum circuit that emulates a self‑attention block.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits used to encode the embedding.
    device : pennylane.Device, optional
        The underlying quantum device.  If ``None`` a default qubit device is
        instantiated.
    """

    def __init__(self, n_qubits: int = 4, device=None):
        self.n_qubits = n_qubits
        # Default to Pennylane's fast simulator
        self.device = device or qml.device("default.qubit", wires=n_qubits)

    def _prepare_params(self, rotation_params: np.ndarray,
                        entangle_params: np.ndarray) -> np.ndarray:
        """Flatten and concatenate the two parameter arrays into the format
        expected by :class:`pennylane.templates.StronglyEntanglingLayers`."""
        return np.concatenate([rotation_params.reshape(-1), entangle_params])

    def _build_qnode(self):
        """Create a parameterised QNode that encodes the input and runs the
        variational layers.  The measurement returns the expectation values of
        Pauli‑Z on each qubit, which are subsequently mapped to a probability
        distribution that plays the role of attention weights."""
        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def circuit(inputs: np.ndarray, params: np.ndarray):
            # Encode the classical input as Z‑rotations
            for i in range(self.n_qubits):
                qml.RZ(inputs[i], wires=i)
            # Apply the variational layer
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """Execute the quantum attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation part of the variational layer
            (shape ``(n_qubits, 3)``).
        entangle_params : np.ndarray
            Parameters for the entanglement part of the variational layer
            (shape ``(n_qubits-1,)``).
        inputs : np.ndarray
            Classical embedding of shape ``(n_qubits,)``.
        shots : int, optional
            Number of shots used when the device supports sampling.  In the
            default simulator this argument is ignored.
        """
        params = self._prepare_params(rotation_params, entangle_params)
        circuit = self._build_qnode()
        # Execute the circuit
        expvals = circuit(inputs, params)
        # Convert expectation values to probabilities in [0, 1]
        probs = (np.array(expvals) + 1.0) / 2.0
        return probs

__all__ = ["SelfAttention"]
