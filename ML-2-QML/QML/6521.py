"""Quantum convolutional neural network with optional classical preprocessing."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


def _conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit convolution primitive used by the QCNN."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc


def _pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit pooling primitive used by the QCNN."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer that operates on all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="ConvolutionalLayer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1])
        qc.append(sub, [i, i + 1])
        qc.barrier()
    return qc


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that reduces the number of qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="PoolingLayer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        sub = _pool_circuit(params[sources.index(src) * 3 : sources.index(src) * 3 + 3], [src, sink])
        qc.append(sub, [src, sink])
        qc.barrier()
    return qc


class QuantumConvolutionalLayer(QuantumCircuit):
    """
    A reusable quantum layer that performs a convolution followed by pooling.
    The number of convolution–pooling repeats (depth) is a configurable
    hyper‑parameter.  The layer exposes its trainable parameters as a
    flat list that can be passed to a variational optimizer.
    """

    def __init__(self, num_qubits: int, depth: int = 2, name: str | None = None):
        super().__init__(num_qubits, name=name or f"QCNNLayer_{depth}")
        self.depth = depth
        self.parameters = []
        for d in range(depth):
            self.append(_conv_layer(num_qubits, f"c{d}"), range(num_qubits))
            self.append(
                _pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)),
                            f"p{d}"),
                range(num_qubits),
            )
            # Record parameters for later extraction
            self.parameters.extend(self._collect_params(f"c{d}"))
            self.parameters.extend(self._collect_params(f"p{d}"))

    def _collect_params(self, prefix: str) -> list[ParameterVector]:
        return [p for p in self.parameters if p.name().startswith(prefix)]


def QCNNHybrid(
    *,
    input_dim: int = 8,
    use_classical_preprocess: bool = True,
    conv_depth: int = 3,
    optimizer_cls: type = COBYLA,
    optimizer_kwargs: dict | None = None,
) -> NeuralNetworkClassifier:
    """
    Create a hybrid QCNN that optionally includes a classical feature map
    before the quantum layers.  The returned object is a
    :class:`~qiskit_machine_learning.algorithms.classifiers.NeuralNetworkClassifier`,
    which can be fitted with the usual scikit‑learn interface.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input feature vector.  Default is 8.
    use_classical_preprocess : bool, optional
        If ``True`` a :class:`~qiskit.circuit.library.ZFeatureMap` is applied
        to the input before the quantum ansatz.  If ``False`` the raw
        feature vector is fed directly.
    conv_depth : int, optional
        Number of convolution‑pooling repeats in the ansatz.  Default is 3.
    optimizer_cls : type, optional
        Optimizer class from :mod:`qiskit_machine_learning.optimizers`.
        Default is :class:`~qiskit_machine_learning.optimizers.COBYLA`.
    optimizer_kwargs : dict, optional
        Keyword arguments forwarded to the optimizer constructor.

    Returns
    -------
    NeuralNetworkClassifier
        A ready‑to‑train hybrid QCNN classifier.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Classical feature map
    feature_map = ZFeatureMap(input_dim) if use_classical_preprocess else None

    # Quantum ansatz
    ansatz = QuantumCircuit(input_dim, name="QCNNAnsatz")
    for d in range(conv_depth):
        ansatz.compose(_conv_layer(input_dim, f"c{d}"), range(input_dim), inplace=True)
        ansatz.compose(
            _pool_layer(list(range(input_dim // 2)), list(range(input_dim // 2, input_dim)),
                        f"p{d}"),
            range(input_dim),
            inplace=True,
        )

    # Observable for a single‑qubit measurement on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (input_dim - 1), 1)])

    # Build the variational quantum neural network
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters if feature_map else [],
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    optimizer = optimizer_cls(**(optimizer_kwargs or {}))

    # Wrap in a scikit‑learn compatible classifier
    return NeuralNetworkClassifier(
        estimator=qnn,
        optimizer=optimizer,
        input_params=feature_map.parameters if feature_map else [],
        weight_params=ansatz.parameters,
    )


__all__ = ["QuantumConvolutionalLayer", "QCNNHybrid"]
