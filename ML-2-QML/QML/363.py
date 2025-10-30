"""QuantumClassifierModel__gen403: Data‑re‑uploading variational classifier with analytic gradients.

The quantum implementation builds a Pennylane QNode that encodes each feature in every
layer, applies a trainable rotation, and entangles neighbouring qubits with CZ gates.
It returns the expectation of the sum of Pauli‑Z observables, which serves as a logit
for binary classification.  Helper functions provide analytic gradients via the
parameter‑shift rule and a single‑step training routine using Adam.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import pennylane as qml
from pennylane.optimize import AdamOptimizer


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[qml.QNode, Iterable[int], Iterable[int], List[qml.PauliOp]]:
    """
    Construct a data‑re‑uploading variational circuit.

    Parameters
    ----------
    num_qubits:
        Number of qubits (equal to the number of input features).
    depth:
        Number of data‑re‑uploading layers.

    Returns
    -------
    circuit:
        A Pennylane QNode that accepts a ``np.ndarray`` of shape ``(num_qubits,)`` and
        returns the expectation value of the sum of Pauli‑Z observables.
    encoding:
        List of qubit indices used for data encoding.
    weights:
        List of variational parameters (flat list of length ``num_qubits * depth``).
    observables:
        List of PauliZ observables for each qubit; the classifier outputs the
        expectation of the sum of these observables as logits.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        idx = 0
        for _ in range(depth):
            # Data encoding
            for qubit in range(num_qubits):
                qml.RX(x[qubit], wires=qubit)
            # Variational rotation
            for qubit in range(num_qubits):
                qml.RY(params[idx + qubit], wires=qubit)
            idx += num_qubits
            # Entanglement
            for qubit in range(num_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
        # Sum of Pauli‑Z expectations as logits
        return sum(qml.expval(qml.PauliZ(qubit)) for qubit in range(num_qubits))

    # Flatten parameter vector
    weights = list(range(num_qubits * depth))
    encoding = list(range(num_qubits))
    observables = [qml.PauliZ(i) for i in range(num_qubits)]

    return circuit, encoding, weights, observables


def parameter_shift_gradients(
    circuit: qml.QNode,
    x_batch: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Compute the analytic gradient of the circuit with respect to all parameters
    using the parameter‑shift rule.

    Parameters
    ----------
    circuit:
        The QNode defined by ``build_classifier_circuit``.
    x_batch:
        Batch of input data of shape ``(batch_size, num_qubits)``.
    params:
        Current variational parameters of shape ``(num_qubits * depth,)``.

    Returns
    -------
    grads:
        Gradient array of shape ``(batch_size, num_params)``.
    """
    shift = np.pi / 2
    grads = np.zeros((x_batch.shape[0], params.size))

    for i in range(params.size):
        shift_vec = np.zeros_like(params)
        shift_vec[i] = shift
        f_plus = np.array([circuit(x, params + shift_vec) for x in x_batch])
        f_minus = np.array([circuit(x, params - shift_vec) for x in x_batch])
        grads[:, i] = (f_plus - f_minus) / 2
    return grads


def train_step(
    circuit: qml.QNode,
    encoding: Iterable[int],
    weights: Iterable[int],
    observables: List[qml.PauliOp],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    lr: float = 0.01,
    optimizer: AdamOptimizer | None = None,
) -> float:
    """
    Perform one gradient‑descent update on a batch of data.

    Parameters
    ----------
    circuit, encoding, weights, observables:
        Objects returned from ``build_classifier_circuit``.
    x_batch:
        Input features of shape ``(batch_size, num_qubits)``.
    y_batch:
        Binary labels of shape ``(batch_size,)``.
    lr:
        Learning rate for the Adam optimizer.
    optimizer:
        Optional pre‑constructed ``AdamOptimizer``; if ``None`` a new one is created.

    Returns
    -------
    loss:
        Mean binary cross‑entropy loss after the update.
    """
    if optimizer is None:
        optimizer = AdamOptimizer(stepsize=lr)

    # Initialise parameters
    params = np.array([0.0] * len(weights))

    # Forward pass: compute logits as sum of Z expectations
    logits = np.array([circuit(x, params) for x in x_batch])
    probs = 1 / (1 + np.exp(-logits))
    loss = -np.mean(
        y_batch * np.log(probs + 1e-10) + (1 - y_batch) * np.log(1 - probs + 1e-10)
    )

    grads = parameter_shift_gradients(circuit, x_batch, params)
    grad_mean = np.mean(grads, axis=0)
    optimizer.step(params, grad_mean)

    return loss


__all__ = ["build_classifier_circuit", "parameter_shift_gradients", "train_step"]
