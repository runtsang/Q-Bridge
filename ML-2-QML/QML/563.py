import pennylane as qml
import numpy as np
from typing import Tuple

class EstimatorQNN:
    """
    Variational quantum regressor using a 2‑qubit circuit.
    Inputs are encoded via Ry rotations, weights are
    applied as Rz rotations, and the expectation of the
    two‑qubit Y⊗Y observable is returned.
    """
    def __init__(self, wires: int = 2, dev: qml.Device | None = None) -> None:
        self.wires = wires
        self.dev = dev or qml.device("default.qubit", wires=self.wires)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Tuple[float, float], weights: Tuple[float, float]) -> float:
            # Feature encoding
            qml.Hadamard(0)
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Parameterized layer
            qml.RZ(weights[0], wires=0)
            qml.RZ(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            # Observable: Y⊗Y
            return qml.expval(qml.PauliY(0) @ qml.PauliY(1))

        self.circuit = circuit

    def __call__(self, inputs: Tuple[float, float], weights: Tuple[float, float]) -> float:
        """
        Predict the regression target for a single sample.
        """
        return float(self.circuit(inputs, weights))

    def predict(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Batch predict by vectorizing the circuit call.
        """
        preds = np.array([self.circuit(tuple(inp), tuple(wt)) for inp, wt in zip(inputs, weights)])
        return preds

__all__ = ["EstimatorQNN"]
