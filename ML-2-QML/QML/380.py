"""
Quantum Convolutional Neural Network implemented with Pennylane.

The architecture follows the original Qiskit design but is re‑expressed
with Pennylane's hybrid‑quantum‑classical interface:
  * A reusable *conv_layer* and *pool_layer* that operate on arbitrary qubit
    groups.
  * A feature‑map based on a Z‑feature map (Pauli Z rotations).
  * A variational ansatz that stacks three conv‑/pool‑pairs.
  * A QNode that returns the expectation value of a single‑qubit Z
    observable, which is interpreted as the network output.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, List


def _conv_circuit(qs: Iterable[int], params: np.ndarray) -> None:
    """
    Two‑qubit convolution unitary used in every conv layer.
    """
    qml.RZ(-np.pi / 2, wires=qs[1])
    qml.CNOT(wires=[qs[1], qs[0]])
    qml.RZ(params[0], wires=qs[0])
    qml.RY(params[1], wires=qs[1])
    qml.CNOT(wires=[qs[0], qs[1]])
    qml.RY(params[2], wires=qs[1])
    qml.CNOT(wires=[qs[1], qs[0]])
    qml.RZ(np.pi / 2, wires=qs[0])


def _pool_circuit(qs: Iterable[int], params: np.ndarray) -> None:
    """
    Two‑qubit pooling unitary – identical to conv without the
    final RZ gate.
    """
    qml.RZ(-np.pi / 2, wires=qs[1])
    qml.CNOT(wires=[qs[1], qs[0]])
    qml.RZ(params[0], wires=qs[0])
    qml.RY(params[1], wires=qs[1])
    qml.CNOT(wires=[qs[0], qs[1]])
    qml.RY(params[2], wires=qs[1])


def _conv_layer(num_qubits: int, start: int) -> List[tuple]:
    """
    Return a list of (qubit pair, parameter slice) tuples for a convolution
    layer.  ``start`` is the index into the global parameter vector.
    """
    pairs = [(i, i + 1) for i in range(0, num_qubits, 2)]
    return [(pair, slice(start + 3 * idx, start + 3 * (idx + 1))) for idx, pair in enumerate(pairs)]


def _pool_layer(sources: List[int], sinks: List[int], start: int) -> List[tuple]:
    """
    Return a list of (qubit pair, parameter slice) tuples for a pooling
    layer.
    """
    return [(pair, slice(start + 3 * idx, start + 3 * (idx + 1))) for idx, pair in enumerate(zip(sources, sinks))]


class QCNN:
    """
    Hybrid quantum‑classical QCNN.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be a power of two, default 8).
    seed : int | None
        Random seed for parameter initialization.
    """

    def __init__(self, n_qubits: int = 8, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.params = np.random.randn(self._total_params()) if seed is None else np.random.RandomState(seed).randn(self._total_params())

    def _total_params(self) -> int:
        """
        Compute the total number of trainable parameters in the ansatz.
        """
        # conv layers: 3 layers with decreasing qubit counts
        conv_params = 3 * (8 + 4 + 2)
        # pool layers: 3 layers with matching qubit counts
        pool_params = 3 * (4 + 2 + 1)
        return conv_params + pool_params

    def __call__(self, features: np.ndarray) -> float:
        """
        Evaluate the QCNN for a single feature vector.
        """
        return self._qnode(features)

    @qml.qnode
    def _qnode(self, x: np.ndarray) -> float:
        # Feature map
        for i, val in enumerate(x):
            qml.RZ(val, wires=i)

        # First conv layer
        idx = 0
        for pair, param_slice in _conv_layer(self.n_qubits, idx):
            _conv_circuit(pair, self.params[param_slice])
        idx += 8 * 3

        # First pool layer
        for pair, param_slice in _pool_layer([0, 1, 2, 3], [4, 5, 6, 7], idx):
            _pool_circuit(pair, self.params[param_slice])
        idx += 4 * 3

        # Second conv layer
        for pair, param_slice in _conv_layer(self.n_qubits // 2, idx):
            _conv_circuit(pair, self.params[param_slice])
        idx += 4 * 3

        # Second pool layer
        for pair, param_slice in _pool_layer([0, 1], [2, 3], idx):
            _pool_circuit(pair, self.params[param_slice])
        idx += 2 * 3

        # Third conv layer
        for pair, param_slice in _conv_layer(self.n_qubits // 4, idx):
            _conv_circuit(pair, self.params[param_slice])
        idx += 2 * 3

        # Third pool layer
        for pair, param_slice in _pool_layer([0], [1], idx):
            _pool_circuit(pair, self.params[param_slice])

        # Output observable
        return qml.expval(qml.PauliZ(0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Batch evaluation – useful for scikit‑learn compatible pipelines.
        """
        return np.array([self(x) for x in X])

    def parameters(self) -> np.ndarray:
        """
        Return a copy of the current parameter vector.
        """
        return self.params.copy()

    def set_parameters(self, new_params: np.ndarray) -> None:
        """
        Update the internal parameters.
        """
        assert new_params.shape == self.params.shape
        self.params = new_params.copy()
