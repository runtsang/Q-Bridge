"""Quantum classifier factory using Pennylane.

Implements an incremental data‑uploading ansatz with a feature map
and a strongly entangling variational layer.  The API matches the
classical counterpart for seamless experimentation.

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
from pennylane import numpy as np


class QuantumClassifierModel:
    """Quantum circuit factory compatible with the classical interface."""

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[object, Iterable[int], Iterable[int], List[qml.operation.Operator]]:
        """
        Construct a variational circuit and return metadata.

        Parameters
        ----------
        num_qubits:
            Number of qubits / input features.
        depth:
            Number of repetitions of the entangling block.

        Returns
        -------
        circuit:
            A Pennylane QNode that accepts a flat parameter vector
            consisting of ``num_qubits`` feature angles followed by
            ``num_qubits * depth * 3`` variational angles.
        encoding:
            Indices of the feature parameters (0.. num_qubits-1).
        weights:
            Indices of the variational parameters (num_qubits..).
        observables:
            List of PauliZ operators to be measured on each qubit.
        """
        dev = qml.device("default.qubit", wires=num_qubits)

        def circuit(*params):
            # Split parameters
            features = params[:num_qubits]
            weights = params[num_qubits:]

            # Feature embedding
            qml.templates.AngleEmbedding(features, wires=range(num_qubits))

            # Variational ansatz: strongly entangling layers
            weights_matrix = np.array(weights).reshape(depth, num_qubits, 3)
            qml.templates.StronglyEntanglingLayers(weights_matrix, wires=range(num_qubits))

            # Return expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        # Wrap as a QNode for automatic differentiation
        qnode = qml.QNode(circuit, dev)

        # Parameter indices
        encoding = list(range(num_qubits))
        # Variational parameters: 3 rotations per qubit per depth
        weights = list(range(num_qubits, num_qubits + num_qubits * depth * 3))

        # Observables
        observables = [qml.PauliZ(i) for i in range(num_qubits)]

        return qnode, encoding, weights, observables

__all__ = ["QuantumClassifierModel"]
