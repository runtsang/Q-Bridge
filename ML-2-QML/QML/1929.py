"""Quantum self‑attention using Pennylane.

The class builds a variational circuit that encodes each input token
into a set of qubits via rotation gates.  Entanglement is introduced
by controlled‑RY gates whose angles are supplied as `entangle_params`.
The attention scores are obtained from the expectation values of
Pauli‑Z on each qubit.  The implementation uses Pennylane’s
parameter‑shift rule for gradients, allowing the block to be
trained end‑to‑end with classical optimisers.

Typical usage::

    sa_q = SelfAttentionEnhancedQuantum(n_qubits=8)
    scores = sa_q.run(rotation_params, entangle_params, inputs, shots=512)

"""

__all__ = ["SelfAttentionEnhancedQuantum"]

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttentionEnhancedQuantum:
    """Variational self‑attention circuit implemented with Pennylane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be ≥ 4 for a meaningful attention block).
    num_layers : int, optional
        Number of entanglement layers; each layer adds a controlled‑RY
        gate between neighbouring qubits.
    device : str | qml.Device, optional
        Pennylane device to run the circuit on.
    """

    def __init__(
        self,
        n_qubits: int,
        num_layers: int = 2,
        device: str | qml.Device = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.device = qml.device(device, wires=n_qubits)

        # Pre‑allocate the qnode
        @qml.qnode(self.device, interface="numpy")
        def circuit(rotation_params, entangle_params, inputs):
            """Internal circuit used by :meth:`run`."""
            # Encode each token by rotating the corresponding qubit.
            for i in range(self.n_qubits):
                # Rotation angles derived from the input embedding
                theta = inputs[i]
                # Gate angles from rotation_params
                rx, ry, rz = rotation_params[3 * i : 3 * i + 3]
                qml.RX(rx, wires=i)
                qml.RY(ry, wires=i)
                qml.RZ(rz, wires=i)
                # Encode the token value
                qml.RY(theta, wires=i)

            # Entanglement layers
            for _ in range(self.num_layers):
                for i in range(self.n_qubits - 1):
                    # Controlled‑RY with angle from entangle_params
                    angle = entangle_params[i]
                    qml.CRY(angle, wires=[i, i + 1])

            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._qnode = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        rotation_params : array, shape (3 * n_qubits,)
            Angles for the Rx, Ry, Rz gates applied to each qubit.
        entangle_params : array, shape (n_qubits - 1,)
            Angles for the controlled‑RY gates in each entanglement layer.
        inputs : array, shape (n_qubits,)
            Token embeddings to be encoded on the qubits.
        shots : int, optional
            Number of measurement shots (ignored when interface='numpy').

        Returns
        -------
        numpy.ndarray
            Expected Pauli‑Z values, interpreted as attention scores.
        """
        # Pennylane’s qnode with interface='numpy' ignores shots;
        # we keep the argument for API compatibility.
        return self._qnode(rotation_params, entangle_params, inputs)
