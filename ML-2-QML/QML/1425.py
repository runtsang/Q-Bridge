"""Variational quantum estimator for EstimatorQNN.

The circuit encodes two classical inputs via RY rotations, applies a
parameterised RZ ansatz, entangles all qubits with a linear CNOT chain,
and measures the expectation of Pauli‑Z on the first qubit.  The
EstimatorQNN function returns a callable that accepts PyTorch tensors
and yields a scalar expectation value, keeping the interface identical
to the classical version.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np


def EstimatorQNN(
    num_qubits: int = 4,
    wires: list[int] | None = None,
) -> callable:
    """Return a variational QNode estimator.

    Args:
        num_qubits: Number of qubits in the ansatz.
        wires: Optional list of wires; defaults to ``range(num_qubits)``.

    Returns:
        A function that takes ``inputs`` and ``weights`` tensors and
        returns the expectation value of Pauli‑Z on the first qubit.
    """
    if wires is None:
        wires = list(range(num_qubits))

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
        # Encode classical inputs
        for i, val in enumerate(inputs):
            qml.RY(val, wires=wires[i])

        # Parameterised ansatz
        for i, w in enumerate(weights):
            qml.RZ(w, wires=wires[i])

        # Entanglement pattern
        for i in range(num_qubits - 1):
            qml.CNOT(wires[wires[i]], wires[wires[i + 1]])

        # Measurement
        return qml.expval(qml.PauliZ(wires[0]))

    def estimator(inputs: np.ndarray, weights: np.ndarray) -> float:
        """Wrapper that accepts PyTorch tensors."""
        inp = np.array(inputs, dtype=np.float64)
        wts = np.array(weights, dtype=np.float64)
        return circuit(inp, wts)

    return estimator


__all__ = ["EstimatorQNN"]
