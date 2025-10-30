import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumSelfAttention:
    """
    Variational quantum circuit implementing a self‑attention‑style
    transformation.  The circuit is parameterised by rotation angles
    and entangling angles and outputs a probability distribution
    over 4 qubits that is interpreted as a key vector.

    The circuit can be executed on a simulator or a real device
    via PennyLane's device interface.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 4).
    device : str or qml.Device, optional
        PennyLane device name or pre‑configured device object.
    """

    def __init__(self, n_qubits: int = 4, device=None):
        self.n_qubits = n_qubits
        if device is None:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        else:
            self.dev = device

        # Define the circuit as a PennyLane QNode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rot, ent):
            for i in range(self.n_qubits):
                qml.RX(rot[3 * i], wires=i)
                qml.RY(rot[3 * i + 1], wires=i)
                qml.RZ(rot[3 * i + 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(ent[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the circuit and return a probability vector derived
        from measurement outcomes.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (n_qubits, 3) containing RX, RY, RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits-1,) containing CRX angles.
        shots : int, optional
            Number of measurement shots (ignored for autograd backend).

        Returns
        -------
        dict
            Mapping from bitstring to counts (if shots specified) or
            expectation values (if using autograd).
        """
        # Flatten parameters for the circuit
        rot = rotation_params.flatten()
        ent = entangle_params
        # Execute the circuit
        result = self.circuit(rot, ent)

        # Convert expectation values to probabilities
        probs = np.abs(result)
        probs /= probs.sum() + 1e-8
        # Return as a simple dict for compatibility with legacy code
        return {f"{i:0{self.n_qubits}b}": val for i, val in enumerate(probs)}

__all__ = ["QuantumSelfAttention"]
