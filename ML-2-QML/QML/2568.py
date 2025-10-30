"""Hybrid quantum encoder for QCNN-based autoencoder."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNNAutoencoder(EstimatorQNN):
    """
    Quantum convolutional encoder that outputs a latent vector of size
    ``latent_dim``.  It is implemented as a QCNN ansatz composed of
    alternating convolution and pooling layers followed by a feature
    map.  The resulting circuit is wrapped in an :class:`EstimatorQNN`
    for easy integration with classical optimizers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map (input dimension).
    latent_dim : int
        Size of the latent vector to be extracted from the circuit.
    """

    def __init__(self, num_qubits: int, latent_dim: int) -> None:
        algorithm_globals.random_seed = 42
        estimator = Estimator()

        feature_map = ZFeatureMap(num_qubits)

        # --- Build QCNN ansatz -------------------------------------------------
        ansatz = QuantumCircuit(num_qubits, name="qc_nn")

        # First convolution
        ansatz.compose(self._conv_layer(num_qubits, "c1"), ansatz.qubits, inplace=True)
        # First pooling
        ansatz.compose(
            self._pool_layer(
                list(range(num_qubits // 2)),
                list(range(num_qubits // 2, num_qubits)),
                "p1",
            ),
            ansatz.qubits,
            inplace=True,
        )
        # Second convolution
        ansatz.compose(
            self._conv_layer(num_qubits // 2, "c2"),
            ansatz.qubits[: num_qubits // 2],
            inplace=True,
        )
        # Second pooling
        ansatz.compose(
            self._pool_layer(
                list(range(num_qubits // 4)),
                list(range(num_qubits // 4, num_qubits // 2)),
                "p2",
            ),
            ansatz.qubits[: num_qubits // 2],
            inplace=True,
        )
        # Third convolution
        ansatz.compose(
            self._conv_layer(num_qubits // 4, "c3"),
            ansatz.qubits[: num_qubits // 4],
            inplace=True,
        )
        # Third pooling
        ansatz.compose(
            self._pool_layer([0], [1], "p3"),
            ansatz.qubits[: num_qubits // 4],
            inplace=True,
        )

        # Observable acting on the first ``latent_dim`` qubits
        obs = SparsePauliOp.from_list(
            [("Z" * latent_dim + "I" * (num_qubits - latent_dim), 1)]
        )

        super().__init__(
            circuit=ansatz.decompose(),
            observables=obs,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _single_conv(params, q1, q2):
        qc = QuantumCircuit(2, name="single_conv")
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="conv")
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._single_conv(params[idx : idx + 3], q1, q2)
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    @staticmethod
    def _single_pool(params, src, snk):
        qc = QuantumCircuit(2, name="single_pool")
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num, name="pool")
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._single_pool(params[idx : idx + 3], src, snk)
            qc.append(sub, [src, snk])
            idx += 3
        return qc

__all__ = ["HybridQCNNAutoencoder"]
