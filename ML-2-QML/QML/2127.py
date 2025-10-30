"""Quantum QCNN implemented in Pennylane with shot‑noise simulation."""

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane import Device
from pennylane import transforms
from pennylane import gates
from pennylane import qml
from typing import Callable


def _conv_layer(qc: qml.QubitDevice, qubits: list[int], params: np.ndarray) -> None:
    """
    Two‑qubit convolution block with a tunable rotation chain.
    Parameters are sliced locally per pair of qubits.
    """
    for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
        idx = i * 3
        qc.rz(-np.pi / 2, q2)
        qc.cx(q2, q1)
        qc.rz(params[idx], q1)
        qc.ry(params[idx + 1], q2)
        qc.cx(q1, q2)
        qc.ry(params[idx + 2], q2)
        qc.cx(q2, q1)
        qc.rz(np.pi / 2, q1)


def _pool_layer(qc: qml.QubitDevice, sources: list[int], sinks: list[int], params: np.ndarray) -> None:
    """
    Two‑qubit pooling block that reduces dimensionality while
    preserving quantum correlations.
    """
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        idx = i * 3
        qc.rz(-np.pi / 2, snk)
        qc.cx(snk, src)
        qc.rz(params[idx], src)
        qc.ry(params[idx + 1], snk)
        qc.cx(src, snk)
        qc.ry(params[idx + 2], snk)


def _feature_map(x: np.ndarray, n_qubits: int) -> None:
    """
    Simple Z‑feature map that encodes input data into qubit phases.
    """
    for i, val in enumerate(x):
        qml.RZ(val, i)


def _quantum_cnn(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Core QCNN circuit combining feature mapping, convolution,
    pooling and a final read‑out observable.
    """
    dev = qml.device("default.qubit", wires=8, shots=1024)  # realistic shot noise

    @qml.qnode(dev, interface="autograd")
    def circuit(x_in: np.ndarray) -> np.ndarray:
        _feature_map(x_in, 8)

        # First convolution + pooling
        _conv_layer(dev, list(range(8)), params[0:24])
        _pool_layer(dev, [0, 1, 2, 3], [4, 5, 6, 7], params[24:39])

        # Second convolution + pooling
        _conv_layer(dev, list(range(4, 8)), params[39:51])
        _pool_layer(dev, [0, 1], [2, 3], params[51:57])

        # Third convolution + pooling
        _conv_layer(dev, list(range(6, 8)), params[57:60])
        _pool_layer(dev, [0], [1], params[60:63])

        # Observable: single‑qubit Z on the first wire
        return qml.expval(qml.PauliZ(0))

    return circuit(x)


def QCNN() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a callable that evaluates the quantum convolutional network
    with explicit shot‑noise simulation. The function signature matches
    the classical QCNN: (x, params) -> output.
    """
    # Parameter vector of appropriate length: 63 free parameters
    # (3 per convolution pair + 3 per pooling pair)
    params = np.random.randn(63, requires_grad=True)

    def forward(x: np.ndarray) -> np.ndarray:
        return _quantum_cnn(params, x)

    return forward


__all__ = ["QCNN"]
