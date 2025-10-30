"""Quantum neural network combining QCNN ansatz with a sampler circuit.

The quantum component builds upon the QCNN ansatz (convolution and pooling
layers) and appends a SamplerQNN‑style subcircuit that produces a two‑output
probability distribution.  Both the expectation‑value QNN and the sampler
share the same feature map and weight parameters, enabling joint training
or sequential evaluation.

The resulting :class:`QCNNGen012QNN` exposes a ``predict`` method that
returns a tuple ``(expectation, probabilities)``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


algorithm_globals.random_seed = 12345  # reproducibility


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used across the QCNN layers."""
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


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary used in pooling layers."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Constructs a convolution layer over all qubits in a pairwise fashion."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
        qc.compose(_conv_circuit(params[3 * i : 3 * i + 3]), [q1, q2], inplace=True)
        qc.barrier()
    return qc


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Constructs a pooling layer that maps sources to sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        qc.compose(_pool_circuit(params[3 * i : 3 * i + 3]), [src, snk], inplace=True)
        qc.barrier()
    return qc


def _sampler_circuit() -> QuantumCircuit:
    """A simple two‑qubit sampler circuit with parameterized rotations."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc


class QCNNGen012QNN:
    """Composite quantum neural network.

    The network first applies the QCNN ansatz (feature map + conv/pool layers)
    and then runs a sampler subcircuit.  The QNN part returns expectation
    values of a single Pauli‑Z observable; the sampler part returns a
    two‑dimensional probability vector over the computational basis.
    """

    def __init__(self) -> None:
        # Feature map
        feature_map = ZFeatureMap(8)

        # QCNN ansatz construction
        ansatz = QuantumCircuit(8, name="QCNN Ansatz")
        # First convolution & pooling
        ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        # Second convolution & pooling
        ansatz.compose(_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        # Third convolution & pooling
        ansatz.compose(_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(_pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        full_circuit = QuantumCircuit(8)
        full_circuit.compose(feature_map, range(8), inplace=True)
        full_circuit.compose(ansatz, range(8), inplace=True)

        # Observable for the expectation output
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Estimator for expectation values
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=full_circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

        # Sampler circuit
        sampler_circuit = _sampler_circuit()
        self.sampler = SamplerQNN(
            circuit=sampler_circuit,
            input_params=ParameterVector("input", 2),
            weight_params=ParameterVector("weight", 4),
            sampler=StatevectorSampler(),
        )

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (expectation, probabilities) for the given input vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (n_samples, 8) for QCNN and (n_samples, 2)
            for the sampler.  The method assumes the caller supplies the
            appropriate slices for each part.
        """
        # Split inputs for QCNN and sampler
        qnn_inputs = inputs[:, :8]
        sampler_inputs = inputs[:, 2:]  # assumes 2‑dimensional sampler input

        exp_vals = self.qnn.predict(qnn_inputs)
        probs = self.sampler.predict(sampler_inputs)

        return exp_vals, probs


def QCNNGen012() -> QCNNGen012QNN:
    """Factory returning a configured :class:`QCNNGen012QNN`."""
    return QCNNGen012QNN()


__all__ = ["QCNNGen012", "QCNNGen012QNN"]
