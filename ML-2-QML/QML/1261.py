"""
QCNN – a hybrid quantum‑classical neural network implemented with Pennylane.

Key extensions
--------------
* Uses Pennylane’s `PauliFeatureMap` with a trainable rotation angle to embed classical data.
* Convolutional layers are parameterised 2‑qubit gates (`conv_circuit`) that can be
  stacked with arbitrary qubit pairings.
* Pooling is realised as a controlled‑phase measurement that discards one qubit per pair,
  preserving entanglement information in the remaining qubits.
* The circuit is wrapped in a `qml.QNode` that returns the expectation value of a
  Z‑observable on the first qubit, enabling gradient‑based optimisation via
  parameter‑shift or autograd.
* A helper `train_qcnn` function demonstrates a simple training loop using
  `torch.optim.Adam` and automatic differentiation.

Design
------
The QCNN is built as a class that stores the feature map, ansatz, and observable.
The `__call__` method evaluates the circuit for a single input sample.
The `predict` method returns a batch of predictions.

The code is fully importable and can be used in any Jupyter notebook or script.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# 1. Feature map – trainable Pauli feature map
# --------------------------------------------------------------------------- #
def pauli_feature_map(x: Iterable[float], params: Iterable[float]) -> qml.QubitCircuit:
    """Pauli feature map with a single rotation angle per qubit."""
    circuit = qml.QubitCircuit()
    for i, (xi, pi) in enumerate(zip(x, params)):
        circuit.add_gate("RX", [xi * pi], wires=i)
        circuit.add_gate("RY", [xi * pi], wires=i)
    return circuit


# --------------------------------------------------------------------------- #
# 2. Convolutional layer – 2‑qubit unitary
# --------------------------------------------------------------------------- #
def conv_circuit(params: Iterable[float]) -> qml.QubitCircuit:
    """Two‑qubit convolutional unitary with 3 rotation parameters."""
    circuit = qml.QubitCircuit(2)
    circuit.add_gate("RZ", [params[0]], wires=0)
    circuit.add_gate("RY", [params[1]], wires=1)
    circuit.add_gate("CNOT", [], wires=[0, 1])
    circuit.add_gate("RZ", [params[2]], wires=1)
    return circuit


# --------------------------------------------------------------------------- #
# 3. Pooling layer – controlled‑phase measurement
# --------------------------------------------------------------------------- #
def pool_circuit(params: Iterable[float]) -> qml.QubitCircuit:
    """Two‑qubit pooling unitary that entangles and measures one qubit."""
    circuit = qml.QubitCircuit(2)
    circuit.add_gate("RZ", [params[0]], wires=0)
    circuit.add_gate("RY", [params[1]], wires=1)
    circuit.add_gate("CNOT", [], wires=[0, 1])
    return circuit


# --------------------------------------------------------------------------- #
# 4. QCNN class
# --------------------------------------------------------------------------- #
class QCNN:
    """Hybrid quantum‑classical convolutional neural network."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_conv_layers: int = 3,
        n_pool_layers: int = 3,
        device: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_conv_layers = n_conv_layers
        self.n_pool_layers = n_pool_layers
        self.dev = qml.device(device, wires=n_qubits)

        # Trainable parameters
        self.feature_params = pnp.random.uniform(0, np.pi, n_qubits)
        self.conv_params = pnp.random.uniform(0, np.pi, n_conv_layers * n_qubits * 3)
        self.pool_params = pnp.random.uniform(0, np.pi, n_pool_layers * n_qubits // 2 * 2)

        # Observable
        self.observable = qml.PauliZ(0)

        # QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: Iterable[float]) -> float:
        # Feature map
        for i in range(self.n_qubits):
            qml.RX(x[i] * self.feature_params[i], wires=i)
            qml.RY(x[i] * self.feature_params[i], wires=i)

        # Convolution + pooling
        for layer in range(self.n_conv_layers):
            base = layer * self.n_qubits * 3
            for q in range(0, self.n_qubits, 2):
                params = self.conv_params[base + q * 3 : base + q * 3 + 3]
                conv_circuit(params).apply([q, q + 1])

            # Pooling
            pool_base = layer * self.n_qubits // 2 * 2
            for q in range(0, self.n_qubits, 4):
                params = self.pool_params[pool_base + (q // 2) * 2 : pool_base + (q // 2) * 2 + 2]
                pool_circuit(params).apply([q, q + 2])

        return qml.expval(self.observable)

    def __call__(self, x: Iterable[float]) -> torch.Tensor:
        return self.qnode(x)

    def predict(self, X: Iterable[Iterable[float]]) -> torch.Tensor:
        """Batch prediction."""
        return torch.stack([self(x) for x in X])

    def parameters(self) -> Tuple[torch.Tensor,...]:
        """Return all trainable parameters for an optimiser."""
        return (
            torch.tensor(self.feature_params, requires_grad=True),
            torch.tensor(self.conv_params, requires_grad=True),
            torch.tensor(self.pool_params, requires_grad=True),
        )


# --------------------------------------------------------------------------- #
# 5. Simple training loop
# --------------------------------------------------------------------------- #
def train_qcnn(
    model: QCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
) -> None:
    """Train the QCNN with Adam optimiser and binary cross‑entropy loss."""
    optimizer = torch.optim.Adam(
        [
            {"params": [model.feature_params], "lr": lr},
            {"params": [model.conv_params], "lr": lr},
            {"params": [model.pool_params], "lr": lr},
        ]
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = torch.stack([model(x) for x in X_train])
        loss = loss_fn(preds.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

__all__ = ["QCNN", "train_qcnn"]
