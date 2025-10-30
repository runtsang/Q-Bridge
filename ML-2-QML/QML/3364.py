"""Quantum‑only QCNN‑Graph model.

This module provides :class:`QCNNGraphHybrid` that implements a quantum
QCNN using Qiskit’s EstimatorQNN.  It mirrors the architecture of the
classical version but exposes only quantum functionality, useful for
experiments that rely solely on quantum forward passes or for
benchmarking against the classical counterpart.

The class offers:
* Construction of a QCNN ansatz with convolution and pooling layers.
* A forward method that returns the expectation value for a given input.
* Utilities to generate random networks, training data, and fidelity‑based
  adjacency graphs on quantum states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
#  Quantum utilities – adapted from the QML seed
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.compose(_conv_circuit(params[i : i + 3]), [i, i + 1], inplace=True)
        qc.barrier()
    for i in range(1, num_qubits - 1, 2):
        qc.compose(_conv_circuit(params[i : i + 3]), [i, i + 1], inplace=True)
        qc.barrier()
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        qc.compose(_pool_circuit(params[params.index(src) : params.index(sink) + 3]), [src, sink], inplace=True)
    return qc


def _build_qcnn(num_qubits: int) -> QuantumCircuit:
    feature_map = ZFeatureMap(num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, range(num_qubits), inplace=True)

    # first conv/pool
    qc.compose(_conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
    qc.compose(_pool_layer([0, 1], [2, 3], "p1"), list(range(num_qubits)), inplace=True)
    # second conv/pool
    qc.compose(_conv_layer(num_qubits // 2, "c2"), list(range(num_qubits // 2)), inplace=True)
    qc.compose(_pool_layer([0], [1], "p2"), list(range(num_qubits // 2)), inplace=True)
    return qc


# --------------------------------------------------------------------------- #
#  Fidelity helpers – adapted from the ML seed
# --------------------------------------------------------------------------- #

def state_fidelity(a: qiskit.quantum_info.Qobj, b: qiskit.quantum_info.Qobj) -> float:
    """Return the absolute squared overlap between two pure state vectors."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qiskit.quantum_info.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Quantum‑only model
# --------------------------------------------------------------------------- #

class QCNNGraphHybrid:
    """
    Quantum‑only QCNN‑Graph model.

    Parameters
    ----------
    num_qubits : int, default 8
        Number of qubits used by the QCNN.
    seed : int | None, default None
        Random seed for reproducibility.
    """

    def __init__(self, num_qubits: int = 8, seed: int | None = None) -> None:
        if seed is not None:
            qiskit.utils.algorithm_globals.random_seed = seed
            np.random.seed(seed)

        self.num_qubits = num_qubits
        self._qcnn_circuit = _build_qcnn(self.num_qubits)
        self._observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        self._estimator = StatevectorEstimator()
        self._qnn = EstimatorQNN(
            circuit=self._qcnn_circuit.decompose(),
            observables=self._observable,
            input_params=self._qcnn_circuit.parameters,
            weight_params=self._qcnn_circuit.parameters,
            estimator=self._estimator,
        )

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return the expectation value of the QCNN for the given input."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self._qnn.predict(x)[0]

    @staticmethod
    def state_fidelity(a: qiskit.quantum_info.Qobj, b: qiskit.quantum_info.Qobj) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qiskit.quantum_info.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """
        Generate a random quantum network architecture and training data.

        Parameters
        ----------
        qnn_arch : Sequence[int]
            Architecture of the QCNN (e.g. [8, 4, 2]).
        samples : int
            Number of training samples to generate.

        Returns
        -------
        Tuple[Sequence[int], List[Qobj], List[Tuple[Qobj, Qobj]], Qobj]
            Architecture, list of unitaries per layer, training dataset, and target unitary.
        """
        from qiskit.quantum_info import random_statevector, random_unitary

        target_unitary = random_unitary(2 ** qnn_arch[-1])
        training_data = [(random_statevector(2 ** qnn_arch[-1]),
                          target_unitary @ random_statevector(2 ** qnn_arch[-1]))
                         for _ in range(samples)]

        unitaries: list[list[qiskit.quantum_info.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: list[qiskit.quantum_info.Qobj] = []
            for output in range(num_outputs):
                op = random_unitary(2 ** (num_inputs + 1))
                if num_outputs > 1:
                    op = qiskit.quantum_info.tensor(op, qiskit.quantum_info.identity(2 ** (num_outputs - 1)))
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

__all__ = ["QCNNGraphHybrid"]
