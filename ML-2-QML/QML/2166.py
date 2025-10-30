"""Quantum variational regressor using Pennylane.

The circuit is a layered ansatz with parametrized rotations and
entanglement gates. The model is compatible with Pennylane's
gradient back‑propagation and can be trained using a simple
optimizer.

The implementation mirrors the original EstimatorQNN but extends it
with:
- multiple qubits proportional to the input feature dimension
- a layered ansatz with Ry and Rz rotations
- a trainable weight register for each rotation
- a configurable number of repeat layers
"""

import pennylane as qml
import numpy as np
from typing import Sequence

def EstimatorQNN(
    input_dim: int = 2,
    num_layers: int = 2,
    weight_shape: Sequence[int] | None = None,
    device_name: str = "default.qubit",
) -> qml.QNode:
    """Return a variational quantum circuit for regression.

    Parameters
    ----------
    input_dim : int
        Number of classical input features; determines the number of qubits.
    num_layers : int
        Number of repeat layers in the ansatz.
    weight_shape : Sequence[int] | None
        Shape of the trainable weight register. If None, inferred as
        (num_layers, input_dim).
    device_name : str
        Pennylane device to use (backend).
    """

    dev = qml.device(device_name, wires=input_dim)

    if weight_shape is None:
        weight_shape = (num_layers, input_dim)

    weight_params = qml.numpy.array(np.random.randn(*weight_shape))

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Data encoding: angle encoding of each feature
        for i, w in enumerate(inputs):
            qml.RY(w, wires=i)

        # Ansatz layers
        for layer in range(num_layers):
            # Rotation gates with trainable weights
            for qubit in range(input_dim):
                qml.RZ(weights[layer, qubit], wires=qubit)
                qml.RY(weights[layer, qubit], wires=qubit)

            # Entanglement layer (nearest‑neighbour)
            for qubit in range(input_dim - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Measurement: expectation of Pauli Z on first qubit
        return qml.expval(qml.PauliZ(0))

    def train(
        self,  # type: ignore
        inputs: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> Sequence[float]:
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_fn = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)

        loss_history: list[float] = []

        for _ in range(epochs):
            def cost(w):
                preds = circuit(inputs, w)
                return loss_fn(preds, targets)

            weight_params, convs = opt.step_and_cost(cost, weight_params)
            loss_history.append(convs)

        return loss_history

    circuit.train = train  # type: ignore[attr-defined]
    return circuit

__all__ = ["EstimatorQNN"]
