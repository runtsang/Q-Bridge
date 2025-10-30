"""Hybrid QCNN – quantum implementation.

This module builds a variational QCNN ansatz that interleaves convolution,
pooling, and self‑attention sub‑circuits.  It mirrors the classical network
defined in ``ml_code`` but leverages Qiskit’s quantum primitives to enable
joint optimisation of quantum weights and classical parameters.

The circuit is constructed for 8 logical qubits.  Each convolutional block
applies a two‑qubit unitary (as in the original QCNN seed).  Pooling reduces
the qubit count, and a custom self‑attention sub‑circuit applies local
rotations followed by controlled‑X entanglement.  The final observable
measures a single Pauli‑Z on the first qubit, emulating a binary classifier.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class QCNNGen574:
    """
    Quantum QCNN with self‑attention sub‑circuits.

    The ansatz is constructed as:
    feature_map -> conv1 -> pool1 -> attn1 -> conv2 -> pool2 -> attn2
    -> conv3 -> pool3 -> attn3
    """

    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = StatevectorEstimator()
        self.backend = Aer.get_backend("statevector_simulator")

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits=8)

        # Build ansatz
        self.ansatz = QuantumCircuit(8, name="Ansatz")
        self.ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        self.ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Append self‑attention sub‑circuits
        self.ansatz.compose(self._attn_layer(8, "a1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._attn_layer(4, "a2"), list(range(4, 8)), inplace=True)
        self.ansatz.compose(self._attn_layer(2, "a3"), list(range(6, 8)), inplace=True)

        # Observable for binary classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Build QNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------ #
    # Helper sub‑circuits
    # ------------------------------------------------------------------ #

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Two‑qubit convolution unit repeated over the qubits."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[3 * i : 3 * (i + 1)])
            qc.append(sub, [qubits[i], qubits[i + 1]])
            qc.barrier()
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Core two‑qubit unitary used in the convolution layers."""
        circ = QuantumCircuit(2)
        circ.rz(-np.pi / 2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        circ.cx(1, 0)
        circ.rz(np.pi / 2, 0)
        return circ

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """Pooling reduces qubit count by discarding sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, snk, p in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[3 * p : 3 * (p + 1)])
            qc.append(sub, [src, snk])
            qc.barrier()
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        circ = QuantumCircuit(2)
        circ.rz(-np.pi / 2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        return circ

    def _attn_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Self‑attention block: local rotations followed by controlled‑X entanglement."""
        qc = QuantumCircuit(num_qubits, name="Self‑Attention Layer")
        rot_params = ParameterVector(f"{param_prefix}_rot", length=num_qubits * 3)
        ent_params = ParameterVector(f"{param_prefix}_ent", length=num_qubits - 1)

        for i in range(num_qubits):
            qc.rx(rot_params[3 * i], i)
            qc.ry(rot_params[3 * i + 1], i)
            qc.rz(rot_params[3 * i + 2], i)

        for i in range(num_qubits - 1):
            qc.crx(ent_params[i], i, i + 1)

        return qc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass using the EstimatorQNN.  The function accepts a batch of
        feature vectors and returns the classifier outputs.
        """
        return self.qnn(inputs).reshape(-1, 1)


def QCNN() -> QCNNGen574:
    """
    Factory that returns a ready‑to‑use quantum QCNN instance.
    """
    return QCNNGen574()


__all__ = ["QCNN", "QCNNGen574"]
