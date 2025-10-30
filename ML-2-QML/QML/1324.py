import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """Variational quantum circuit emulating a fully connected layer.

    The circuit uses a parameterized ansatz with alternating single-qubit rotations
    and CNOT entangling gates. It returns the expectation value of the Pauli-Z
    operator on the first qubit, which can be interpreted as a classical
    activation output.
    """

    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(params):
            for w in range(self.n_qubits):
                qml.RY(params[w], wires=w)
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for a list of parameter values.

        Parameters
        ----------
        thetas : Iterable[float]
            List of theta values (one per qubit).

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value.
        """
        params = np.array(thetas, dtype=np.float32)
        expectation = self.circuit(params)
        return np.array([expectation])
