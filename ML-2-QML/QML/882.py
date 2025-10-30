"""EstimatorQNN – a quantum neural network estimator built with PennyLane."""
from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

class EstimatorQNN:
    """
    Quantum regression model implemented as a variational circuit.

    Features:
    * 2‑qubit circuit with RX input encoding and RY/RZ variational angles.
    * CNOT entanglement between the qubits.
    * Measurement of the Y⊗Y observable.
    * Parameter‑shift gradients via PennyLane's autograd interface.
    * Adam optimiser for training.
    """
    def __init__(self,
                 num_qubits: int = 2,
                 init_params: np.ndarray | None = None) -> None:
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.wires = list(range(num_qubits))

        # Initialise parameters: 3 per qubit (RY, RZ, plus an extra for future layers)
        if init_params is None:
            self.params = pnp.random.randn(num_qubits * 3)
        else:
            self.params = init_params

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, inputs):
            # Input encoding
            for i, wire in enumerate(self.wires):
                qml.RX(inputs[i], wire)
            # Variational layers
            for i, wire in enumerate(self.wires):
                qml.RY(params[i], wire)
                qml.RZ(params[i + self.num_qubits], wire)
            # Entanglement
            qml.CNOT(self.wires[0], self.wires[1])
            # Observable
            return qml.expval(qml.PauliY(self.wires[0]) @ qml.PauliY(self.wires[1]))

        self.circuit = circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum circuit.
        :param inputs: shape (batch, num_qubits)
        :return: predictions shape (batch,)
        """
        preds = [self.circuit(self.params, x) for x in inputs]
        return np.array(preds)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 200,
              lr: float = 0.01) -> None:
        """
        Train the circuit parameters using Adam optimizer.
        """
        opt = AdamOptimizer(lr)
        for _ in range(epochs):
            def loss_fn(params):
                preds = self.predict(X)
                return np.mean((preds - y) ** 2)
            self.params = opt.step(loss_fn, self.params)

def EstimatorQNN() -> EstimatorQNN:
    """Factory returning a ready‑to‑train EstimatorQNN quantum instance."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
