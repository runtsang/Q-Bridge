import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Quantum variational fully‑connected layer implemented with Pennylane.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the circuit.
    n_layers : int, default 2
        Number of variational layers (each layer consists of single‑qubit rotations
        followed by a full‑SWAP entanglement).
    device : str or qml.Device, default "default.qubit"
        Pennylane device used for simulation or real hardware.
    shots : int, default 1024
        Number of shots for expectation evaluation.
    """

    def __init__(self,
                 n_qubits: int = 1,
                 n_layers: int = 2,
                 device: str | qml.Device = "default.qubit",
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # params shape: (n_layers, n_qubits, 3)  -> rotations (rx, ry, rz)
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                # Full‑SWAP entanglement across all qubits
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the variational circuit for a batch of parameter sets.

        Parameters
        ----------
        thetas : np.ndarray
            Array of shape (batch, n_layers * n_qubits * 3) containing rotation angles.

        Returns
        -------
        np.ndarray
            Array of shape (batch, n_qubits) with expectation values of Pauli‑Z.
        """
        thetas = np.asarray(thetas, dtype=np.float32)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        batch_expectations = []
        for theta_batch in thetas:
            # Reshape to (n_layers, n_qubits, 3)
            params = theta_batch.reshape(self.n_layers, self.n_qubits, 3)
            exp_vals = self.circuit(params)
            batch_expectations.append(exp_vals)
        return np.vstack(batch_expectations)

__all__ = ["FullyConnectedLayer"]
