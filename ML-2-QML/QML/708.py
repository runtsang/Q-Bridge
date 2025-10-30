import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Quantum fully connected layer implemented as a variational ansatz.
    Supports arbitrary number of qubits (features) and a trainable
    parameter vector that is mapped to rotation angles.
    """
    def __init__(self,
                 n_qubits: int,
                 dev: qml.Device = None,
                 wires: Iterable[int] = None,
                 layers: int = 2,
                 entanglement: str = "full"):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.wires = wires or list(range(n_qubits))
        self.dev = dev or qml.device("default.qubit", wires=self.wires)
        self.theta = np.random.randn(self._num_params())
        self._build_circuit()

    def _num_params(self):
        # Each layer has n_qubits rotations
        return self.layers * self.n_qubits

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Encode input as basis states (one-hot)
            for i, w in enumerate(self.wires):
                qml.BasisState([1 if i == j else 0 for j in range(self.n_qubits)], w)
            # Variational layers
            for l in range(self.layers):
                for i, w in enumerate(self.wires):
                    qml.RY(params[l * self.n_qubits + i], w)
                if self.entanglement == "full":
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(self.wires[i], self.wires[i + 1])
                elif self.entanglement == "circular":
                    qml.CNOT(self.wires[-1], self.wires[0])
            return qml.expval(qml.PauliZ(self.wires[-1]))
        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit with the provided parameter vector.
        Returns a NumPy array containing the expectation value.
        """
        params = np.asarray(thetas, dtype=np.float64)
        if params.size!= self._num_params():
            raise ValueError(f"Expected {self._num_params()} parameters, got {params.size}")
        self.theta = params
        expectation = self._circuit(params)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation w.r.t. the parameters
        using the autograd interface.
        """
        params = np.asarray(thetas, dtype=np.float64)
        if params.size!= self._num_params():
            raise ValueError(f"Expected {self._num_params()} parameters, got {params.size}")
        return qml.grad(self._circuit)(params)

__all__ = ["FCL"]
