"""Hybrid QCNN‑Graph model combining classical CNN, quantum QCNN, and graph fidelity.

The module defines :class:`QCNNGraphHybrid` which offers:
* A classical feed‑forward network mirroring the QCNN seed.
* A quantum QCNN built on Qiskit’s EstimatorQNN.
* Utilities to generate synthetic data, random networks, and fidelity‑based adjacency graphs.
* Methods to run either the classical or quantum forward pass and to compare the two representations.

The class is intentionally lightweight and focuses on the model structure rather than training loops.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
#  Classical utilities – adapted from the ML seed
# --------------------------------------------------------------------------- #

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight list, training data and target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
#  Quantum utilities – adapted from the QML seed
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN."""
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
    """Wrap the conv_circuit in a multi‑qubit pattern."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.compose(_conv_circuit(params[i : i + 3]), [i, i + 1], inplace=True)
        qc.barrier()
    # second pass for odd‑even pairs
    for i in range(1, num_qubits - 1, 2):
        qc.compose(_conv_circuit(params[i : i + 3]), [i, i + 1], inplace=True)
        qc.barrier()
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    """Construct a pooling layer that reduces the number of qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        qc.compose(_pool_circuit(params[params.index(src) : params.index(sink) + 3]), [src, sink], inplace=True)
    return qc


def _build_qcnn(num_qubits: int) -> QuantumCircuit:
    """Return a full QCNN ansatz with the same depth as the seed."""
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
#  Fidelity helpers – shared between ML and QML
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap between two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from pairwise fidelities."""
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
#  Hybrid model
# --------------------------------------------------------------------------- #

class QCNNGraphHybrid:
    """
    Hybrid QCNN‑Graph model that combines classical CNN, quantum QCNN, and graph fidelity.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture for the classical feed‑forward network (e.g. [8, 16, 8, 1]).
    num_qubits : int, default 8
        Number of qubits used by the quantum QCNN.
    seed : int | None, default None
        Random seed for reproducibility.
    """

    def __init__(self, arch: Sequence[int], num_qubits: int = 8, seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            qiskit.utils.algorithm_globals.random_seed = seed

        self.arch = list(arch)
        self.num_qubits = num_qubits

        # Classical network
        layers = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.pop()  # remove last activation
        self.classical = nn.Sequential(*layers)

        # Quantum QCNN
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

    # --------------------------------------------------------------------- #
    #  Classical forward pass
    # --------------------------------------------------------------------- #
    def forward_classical(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the output of the classical CNN."""
        return torch.sigmoid(self.classical(x))

    # --------------------------------------------------------------------- #
    #  Quantum forward pass
    # --------------------------------------------------------------------- #
    def forward_quantum(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Evaluate the QCNN on the given input.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input vector of shape (n_features,).
        Returns
        -------
        np.ndarray
            Expectation value of the observable.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self._qnn.predict(x)[0]

    # --------------------------------------------------------------------- #
    #  Graph utilities
    # --------------------------------------------------------------------- #
    def fidelity_graph(self, activations: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """
        Build a weighted adjacency graph from the given activations.

        Parameters
        ----------
        activations : Sequence[torch.Tensor]
            List of activation vectors (e.g. from the classical network).
        threshold : float
            Fidelity threshold for edge creation.
        secondary : float | None, optional
            Secondary threshold for lower‑weight edges.
        """
        return fidelity_adjacency(activations, threshold, secondary=secondary)

    # --------------------------------------------------------------------- #
    #  Utility helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        return random_training_data(weight, samples)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = ["QCNNGraphHybrid"]
