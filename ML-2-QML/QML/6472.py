"""Quantum fully connected layer implemented as a variational circuit using PennyLane."""

import json
import numpy as np
import pennylane as qml
from typing import Iterable


class FullyConnectedLayer:
    """
    Quantum neural network layer that maps a list of parameters (thetas) to a single expectation value.
    The circuit uses a parameterized rotation layer followed by entanglement and a measurement of PauliZ.
    """

    def __init__(self, n_qubits: int = 2, device_name: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)

        @qml.qnode(self.device, interface="autograd")
        def circuit(thetas):
            # Apply a rotation layer with the provided parameters
            for i, wire in enumerate(range(n_qubits)):
                qml.RX(thetas[i], wires=wire)
                qml.RY(thetas[i], wires=wire)
                qml.RZ(thetas[i], wires=wire)
            # Entanglement (CNOT chain)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in `thetas` and return the expectation values.
        """
        thetas = np.asarray(thetas, dtype=np.float64)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        expectations = []
        for params in thetas:
            exp_val = self._circuit(params)
            expectations.append(exp_val)
        return np.array(expectations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method for batch inference.
        """
        return self.run(X)

    def save(self, path: str) -> None:
        """
        Save the device configuration and circuit parameters to a JSON file.
        """
        meta = {
            "n_qubits": self.n_qubits,
            "device_name": self.device.name,
            "shots": self.device.shots,
        }
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "FullyConnectedLayer":
        """
        Load a previously saved quantum layer.
        """
        with open(f"{path}.json", "r") as f:
            meta = json.load(f)
        return cls(
            n_qubits=meta["n_qubits"],
            device_name=meta["device_name"],
            shots=meta["shots"],
        )


__all__ = ["FullyConnectedLayer"]
