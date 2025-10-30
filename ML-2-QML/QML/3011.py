"""Quantum implementation of the QCNN regression/classification hybrid.

This module builds on the QCNN construction from the original
reference and augments it with a regression head.  The same feature
map and convolution‑pooling ansatz are used for both tasks; the
difference lies in the observable used by the EstimatorQNN and the
post‑processing of the expectation value.

The class :class:`QCNNRegressionQNN` exposes a ``predict`` method that
returns either a probability in ``[0,1]`` (classification) or a
continuous value (regression).  The underlying quantum circuit is
compatible with the Qiskit Aer simulator and can be run on real
hardware via the Qiskit runtime API.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Dataset utilities – identical to the classical counterpart
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and corresponding labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits / dimensionality of the input.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape ``(samples, 2**num_wires)``.
    labels : np.ndarray
        Real labels of shape ``(samples,)``.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


# --------------------------------------------------------------------------- #
# Quantum building blocks
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN ansatz."""
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
    """Construct a convolution layer that applies ``conv_circuit`` to
    adjacent qubit pairs in a zig‑zag pattern."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps ``sources`` to ``sinks``."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# QCNN regression/classification hybrid
# --------------------------------------------------------------------------- #
class QCNNRegressionQNN:
    """Hybrid quantum neural network capable of classification or regression.

    Parameters
    ----------
    mode : str, optional
        ``'classify'`` (default) or ``'regress'``.  The observable and
        post‑processing are chosen accordingly.
    """

    def __init__(self, mode: str = "classify") -> None:
        algorithm_globals.random_seed = 12345
        self.mode = mode

        # Feature map: encode 8‑dimensional classical data
        self.feature_map = ZFeatureMap(8)

        # Build the ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolution & pooling
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

        # Second convolution & pooling
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Third convolution & pooling
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable choice
        if mode == "classify":
            # Single‑qubit Z observable on the last qubit
            observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        else:  # regress
            # Multi‑qubit Z observable to capture a larger feature space
            observable = SparsePauliOp.from_list([("Z" * 8, 1)])

        # Estimator backend
        estimator = Estimator()

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray of shape (batch, 8)
            Classical feature vectors.

        Returns
        -------
        preds : np.ndarray
            For classification: probabilities in ``[0,1]``.
            For regression: continuous values.
        """
        preds = self.qnn.predict(inputs).real.squeeze(-1)
        if self.mode == "classify":
            return 1 / (1 + np.exp(-preds))  # sigmoid
        else:
            return preds

    def set_mode(self, mode: str) -> None:
        """Switch between classification and regression."""
        if mode not in ("classify", "regress"):
            raise ValueError("mode must be either 'classify' or'regress'")
        self.mode = mode
        # Re‑build the QNN with the appropriate observable
        self.__init__(mode)

__all__ = [
    "QCNNRegressionQNN",
    "generate_superposition_data",
]
