"""Quantum QCNN model implementing convolution and pooling with variational parameters.

The class :class:`QCNNModel` wraps an EstimatorQNN and exposes a ``forward`` method
that accepts a NumPy array of shape (batch, features) and returns the
probability of the positive class.  The interface mirrors the classical
:class:`~.QCNNModel`, enabling direct comparison of performance metrics.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap


# Helper functions from the original QCNN quantum implementation
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that applies ``conv_circuit`` to adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block that reduces entanglement."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps a set of source qubits to a set of sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


def build_qcnn_ansatz(num_qubits: int = 8) -> QuantumCircuit:
    """Construct the QCNN ansatz: feature map + 3 conv/pool stages."""
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # Stage 1
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(
        pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"),
        range(num_qubits),
        inplace=True,
    )

    # Stage 2
    ansatz.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
    ansatz.compose(
        pool_layer(list(range(num_qubits // 4)), list(range(num_qubits // 4, num_qubits // 2)), "p2"),
        range(num_qubits // 2, num_qubits),
        inplace=True,
    )

    # Stage 3
    ansatz.compose(conv_layer(num_qubits // 4, "c3"), range(num_qubits - num_qubits // 4, num_qubits), inplace=True)
    ansatz.compose(pool_layer([num_qubits - 1], [num_qubits - 2], "p3"),
                   range(num_qubits - num_qubits // 4, num_qubits),
                   inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit


def build_observables(num_qubits: int = 8) -> list[SparsePauliOp]:
    """Return a list of Z observables on each qubit, suitable for a binary output."""
    return [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]


class QCNNModel:
    """Quantum implementation of the QCNN architecture.

    The class wraps an EstimatorQNN and exposes a ``forward`` method that
    accepts a NumPy array of shape (batch, features) and returns the
    probability of the positive class.  The interface mirrors the
    classical :class:`~.QCNNModel`, enabling direct comparison of
    performance metrics.
    """
    def __init__(self, num_qubits: int = 8, estimator: Estimator | None = None):
        self.num_qubits = num_qubits
        self.estimator = estimator or Estimator()
        circuit = build_qcnn_ansatz(num_qubits)
        observables = build_observables(num_qubits)
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observables,
            input_params=ZFeatureMap(num_qubits).parameters,
            weight_params=circuit.parameters,
            estimator=self.estimator,
        )
        # Expose weight sizes and observables for compatibility
        self.weight_sizes = [len(circuit.parameters)]
        self.observables = [op.to_label() for op in observables]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the quantum circuit on the given inputs.

        Parameters
        ----------
        inputs
            Array of shape (batch, num_qubits) containing feature values.
        Returns
        -------
        probs
            Array of shape (batch,) with the probability of the positive class
            (obtained from the first Pauli‑Z observable).
        """
        results = self.qnn.predict(inputs)
        # The first observable corresponds to the positive class
        return results[:, 0]


def QCNN() -> QCNNModel:
    """Factory returning a configured :class:`QCNNModel` instance."""
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
