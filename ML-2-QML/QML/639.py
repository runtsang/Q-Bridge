"""
EstimatorQNN – Variational quantum regression model using Pennylane.

Key features
------------
* A custom variational circuit with alternating rotation and entanglement layers.
* Parameter‑shift gradient for analytic derivatives.
* Training loop that mirrors the classical counterpart.
* Supports both CPU and GPU back‑ends via Pennylane's `default.quad` or `default.qubit`.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, Tuple, Dict, Any
import numpy as np

__all__ = ["EstimatorQNN", "train_qnn", "evaluate_qnn", "make_synthetic_data"]


def EstimatorQNN(
    num_qubits: int = 4,
    layers: int = 3,
    device: str = "default.qubit",
    seed: int | None = None,
) -> qml.QNode:
    """
    Build a variational circuit for regression.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    layers : int
        Number of rotation-entanglement blocks.
    device : str
        Pennylane device name. ``default.qubit`` or ``default.quad``.
    seed : int, optional
        Random seed for weight initialization.

    Returns
    -------
    qml.QNode
        Callable variational circuit that returns a scalar expectation value.
    """
    dev = qml.device(device, wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
        # Encode inputs as rotation angles
        for i, w in enumerate(inputs):
            qml.RX(w, wires=i % num_qubits)

        # Variational layers
        for layer in range(layers):
            for q in range(num_qubits):
                qml.RY(weights[layer, q], wires=q)
            # Entanglement
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[num_qubits - 1, 0])  # ring topology

        # Observable: PauliZ on first qubit
        return qml.expval(qml.PauliZ(0))

    # Initialise weights
    rng = np.random.default_rng(seed)
    init_weights = rng.normal(0, np.pi, size=(layers, num_qubits))
    circuit.weights = init_weights

    return circuit


def train_qnn(
    circuit: qml.QNode,
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 200,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the variational circuit using gradient descent.

    Parameters
    ----------
    circuit : qml.QNode
        Variational circuit with attached weights.
    X : np.ndarray
        Input features (num_samples, num_features).
    y : np.ndarray
        Target values (num_samples,).
    learning_rate : float
        Learning rate for the optimizer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini‑batch size.
    verbose : bool
        If True, prints loss per epoch.

    Returns
    -------
    dict
        Dictionary containing final loss and weight history.
    """
    opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    weights = circuit.weights
    history = []

    for epoch in range(1, epochs + 1):
        # Shuffle indices
        idx = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[idx], y[idx]
        epoch_loss = 0.0

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb, yb = X_shuffled[start:end], y_shuffled[start:end]
            # Loss: mean squared error
            def loss_fn(w):
                preds = np.array([circuit(x, w) for x in xb])
                return np.mean((preds - yb) ** 2)

            weights, loss = opt.step_and_cost(loss_fn, weights)
            epoch_loss += loss * len(xb)

        epoch_loss /= len(X)
        history.append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")

    circuit.weights = weights
    return {"loss": history[-1], "history": history, "weights": weights}


def evaluate_qnn(
    circuit: qml.QNode,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute mean squared error on a dataset.

    Parameters
    ----------
    circuit : qml.QNode
        Trained circuit.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Targets.

    Returns
    -------
    float
        Mean squared error.
    """
    preds = np.array([circuit(x, circuit.weights) for x in X])
    return np.mean((preds - y) ** 2)


def make_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 2,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy regression dataset (same as classical counterpart).

    Returns
    -------
    X : np.ndarray
        Features shape (num_samples, num_features).
    y : np.ndarray
        Targets shape (num_samples,).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, num_features))
    y = np.sum(X**2, axis=1) + 0.5 * np.sum(X, axis=1)
    y += noise_std * rng.standard_normal(num_samples)
    return X, y
