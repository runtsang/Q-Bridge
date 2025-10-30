import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class FullyConnectedLayerExtended:
    """
    A quantum analogue of the classical fully‑connected layer.  The circuit
    uses a parameter‑shift ansatz with two qubits, entanglement, and a
    measurement of the Pauli‑Z expectation value.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the ansatz.  Default 2.
    device : str or pennylane.Device
        Backend device.  Default to a local qasm simulator.
    shots : int
        Number of shots for expectation estimation.  Default 1024.
    """
    def __init__(self, n_qubits: int = 2, device: str | qml.Device = "default.qubit", shots: int = 1024) -> None:
        if isinstance(device, str):
            self.dev = qml.device(device, wires=n_qubits, shots=shots)
        else:
            self.dev = device

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas: np.ndarray):
            # Prepare each qubit with a parameterised Ry rotation
            for w, theta in enumerate(thetas):
                qml.RY(theta, wires=w)
            # Entangle with a CNOT chain
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # Measure Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the expectation value for a list of parameter values.
        The routine accepts a 1‑D iterable of thetas; each theta is used
        to rotate the corresponding qubit.

        Parameters
        ----------
        thetas : Iterable[float]
            List of angles (in radians) for each qubit.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the mean expectation over the
            provided angles.
        """
        theta_arr = np.array(list(thetas), dtype=np.float32)
        # Ensure the correct shape (n_qubits,)
        if theta_arr.size!= self.dev.num_wires:
            raise ValueError(f"Expected {self.dev.num_wires} thetas, got {theta_arr.size}")
        exp = self.circuit(theta_arr)
        return np.array([exp])

__all__ = ["FullyConnectedLayerExtended"]
