import pennylane as qml
import numpy as np
from typing import Iterable

class QuantumFullyConnectedLayer:
    """
    Variational quantum circuit that emulates a single‑qubit fully connected
    layer.  The circuit can be executed on a simulator or on a real device
    through Pennylane's device abstraction.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.  The first qubit holds the
        measurement outcome that is returned as the layer output.
    device_str : str
        Pennylane device identifier (e.g. ``default.qubit`` or
        ``qiskit.qasm_simulator``).
    shots : int
        Number of shots for stochastic evaluation.
    """
    def __init__(self, n_qubits: int = 4, device_str: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_str, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray):
            for qubit in range(n_qubits):
                qml.RY(params[qubit], wires=qubit)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value as a NumPy array, mirroring the classical API.
        """
        params = np.array(list(thetas), dtype=float)
        if params.size!= self.n_qubits:
            raise ValueError("Parameter list length must match ``n_qubits``.")
        expectation = self.circuit(params)
        return np.array([expectation])

    def train(self, X: np.ndarray, y: np.ndarray,
              lr: float = 0.01, epochs: int = 100) -> np.ndarray:
        """
        Very small training loop that optimises the parameters to minimise
        mean‑squared‑error on a toy dataset.  The routine is intentionally
        minimal so that it can be used as a drop‑in for experiments.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        params = np.random.randn(self.n_qubits)

        for _ in range(epochs):
            def cost(p):
                preds = np.array([self.circuit(p) for _ in X])
                return ((preds - y)**2).mean()
            params = opt.step(cost, params)

        return params
