"""Enhanced quantum classifier factory with variational ansatz, skip connections, and balanced readout."""

from __future__ import annotations

from typing import Iterable, Tuple

import pennylane as qml
import pennylane.numpy as np
import torch

__all__ = ["build_classifier_circuit"]

def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             *,
                             skip_connection: bool = True,
                             entangle: bool = True,
                             readout: str = "Z") -> Tuple[qml.QNode,
                                                            Iterable,
                                                            Iterable,
                                                            list[tuple[str, float]]]:
    """
    Construct a variational quantum circuit for binary classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of ansatz layers.
    skip_connection : bool, optional
        If ``True`` a unitary that mimics a residual connection is applied
        between layers by re‑applying the previous rotation parameters.
    entangle : bool, optional
        Whether to apply entangling CZ gates between neighbouring qubits.
    readout : str, optional
        Pauli string to read out.  ``"Z"`` returns the expectation of Z on
        each qubit, ``"ZZ"`` returns a pairwise observable, etc.

    Returns
    -------
    qnode : qml.QNode
        The callable quantum node ready for training.
    encoding : Iterable
        Parameter vector for data encoding (Rx rotations).
    weights : Iterable
        Parameter vector for the variational ansatz.
    observables : list[tuple[str, float]]
        List of tuples describing the readout observables and a scaling
        factor that can be used in a balanced loss.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Data encoding
        for i in range(num_qubits):
            qml.RX(inputs[i], wires=i)

        # Variational layers
        for d in range(depth):
            for i in range(num_qubits):
                qml.RY(weights[d * num_qubits + i], wires=i)
            if entangle:
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # Skip connection: re‑apply previous rotation parameters
            if skip_connection and d > 0:
                for i in range(num_qubits):
                    qml.RY(weights[(d - 1) * num_qubits + i], wires=i)

        # Readout
        if readout == "Z":
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        elif readout == "ZZ":
            return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                    for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        else:
            raise ValueError(f"Unsupported readout {readout}")

    # Dummy encoding and weight names for compatibility
    encoding = [f"x{i}" for i in range(num_qubits)]
    weights = [f"theta{d * num_qubits + i}" for d in range(depth) for i in range(num_qubits)]

    observables = [(readout, 1.0)]

    return circuit, encoding, weights, observables
