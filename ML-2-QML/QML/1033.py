import pennylane as qml
import numpy as np

class FCL:
    """
    Parameterized quantum circuit that emulates a fully connected layer.
    Implements a simple Ansatz with Hadamard initialization and RY rotations,
    measuring the expectation value of PauliZ on the first qubit.
    Provides analytic gradients via Pennylane's autograd interface.
    """
    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit", shots: int = 1024):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        dev_name : str
            Pennylane device name (e.g., "default.qubit", "qiskit.aer").
        shots : int
            Number of shots for stochastic execution; ignored when using autograd.
        """
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            qml.Hadamard(wires=range(n_qubits))
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit with the supplied parameters.
        Returns a 1‑D array containing the expectation value.
        """
        return np.array([self.circuit(thetas)])

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the expectation value with respect to the parameters
        using Pennylane's automatic differentiation.
        Returns a 1‑D array of gradients.
        """
        return np.array([qml.grad(self.circuit)(thetas)])

__all__ = ["FCL"]
