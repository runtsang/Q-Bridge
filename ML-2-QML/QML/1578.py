import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Quantum variational circuit that emulates a fully‑connected layer.
    Parameters
    ----------
    n_qubits : int
        Number of qubits (should match the dimensionality of the input).
    dev : str | qml.Device, default 'default.qubit'
        Pennylane device used for simulation or real hardware.
    """

    def __init__(self, n_qubits: int = 1, dev: str | qml.Device = 'default.qubit'):
        self.n_qubits = n_qubits
        if isinstance(dev, str):
            self.dev = qml.device(dev, wires=n_qubits, shots=1024)
        else:
            self.dev = dev

        @qml.qnode(self.dev, interface='autograd')
        def circuit(thetas):
            # Encode each theta into an Ry rotation on each qubit
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)
            # Simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # Measurement of the first qubit's Z expectation
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the supplied angles and return the expectation value.
        Thetas must be an array of length `n_qubits`.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} thetas, got {len(thetas)}.")
        expectation = self.circuit(thetas)
        return np.array([expectation])

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the angles
        using the parameter‑shift rule built into Pennylane.
        """
        return qml.grad(self.circuit)(thetas)

__all__ = ["FullyConnectedLayer"]
