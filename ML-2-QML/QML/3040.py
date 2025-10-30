"""Quantum implementation of the hybrid Graph‑QCNN architecture.

The QML counterpart uses Qiskit circuits that mirror the
classical model: a feature map, a stack of convolutional and
pooling layers, and a final observable.  The `HybridGraphQCNNQML`
class exposes a `predict` method that returns expectation values
for a batch of classical inputs."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info.random import random_unitary
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# ---------- Utility functions ----------

def state_fidelity(a: qi.Statevector, b: qi.Statevector) -> float:
    """Squared overlap of two pure states."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency(
    states: Sequence[qi.Statevector], threshold: float,
    *, secondary: float | None = None, secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(unitary: qi.Unitary, samples: int):
    dataset = []
    num_qubits = unitary.num_qubits
    for _ in range(samples):
        state = qi.Statevector.random(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = qi.Unitary(random_unitary(qnn_arch[-1]))
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qi.Unitary]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qi.Unitary] = []
        for _ in range(num_outputs):
            op = qi.Unitary(random_unitary(num_inputs + 1))
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qi.Unitary]],
    layer: int,
    input_state: qi.Statevector,
) -> qi.Statevector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Prepare input state with ancilla qubits initialized to |0⟩
    ancilla = qi.Statevector.from_label("0" * num_outputs)
    state = qi.Statevector(np.kron(input_state.data, ancilla.data))

    # Apply the layer unitary
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = layer_unitary @ gate
    new_state = layer_unitary @ state

    # Partial trace over ancilla qubits
    keep = list(range(num_inputs))
    return qi.Statevector(new_state.data, dims=[2] * len(keep))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qi.Unitary]],
    samples: Iterable[Tuple[qi.Statevector, qi.Statevector]],
) -> List[List[qi.Statevector]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

# ---------- Convolution & pooling primitives (QCNN style) ----------

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[(i // 2) * 3 : (i // 2 + 1) * 3])
        qc.append(sub, [i, i + 1])
    return qc


def pool_layer(
    sources: List[int], sinks: List[int], param_prefix: str
) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for idx, (src, sink) in enumerate(zip(sources, sinks)):
        sub = pool_circuit(params[idx * 3 : (idx + 1) * 3])
        qc.append(sub, [src, sink])
    return qc

# ---------- Hybrid quantum model ----------

class HybridGraphQCNNQML:
    """Quantum counterpart of `HybridGraphQCNN` using Qiskit circuits."""

    def __init__(self, arch: Sequence[int], conv_layers: int = 3) -> None:
        self.arch = list(arch)
        self.conv_layers = conv_layers

        total_qubits = sum(arch)
        self.circuit = QuantumCircuit(total_qubits)

        # Feature map
        self.feature_map = ZFeatureMap(total_qubits)
        self.circuit.compose(self.feature_map, range(total_qubits), inplace=True)

        # Build convolution + pooling stack
        qubits = list(range(total_qubits))
        for layer in range(conv_layers):
            self.circuit.compose(conv_layer(len(qubits), f"c{layer+1}"),
                                 qubits, inplace=True)
            sink_qubits = qubits[len(qubits) // 2 :]
            self.circuit.compose(
                pool_layer(qubits[: len(qubits) // 2], sink_qubits, f"p{layer+1}"),
                qubits,
                inplace=True,
            )
            qubits = sink_qubits

        # Observable (Z on first qubit)
        self.observable = qi.SparsePauliOp.from_list(
            [("Z" + "I" * (total_qubits - 1), 1)]
        )

        # Estimator and QNN wrapper
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            estimator=self.estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of classical inputs."""
        return self.qnn.predict(X)

    @staticmethod
    def random_model(
        arch: Sequence[int], samples: int = 1000, conv_layers: int = 3
    ) -> tuple["HybridGraphQCNNQML", List[Tuple[qi.Statevector, qi.Statevector]]]:
        _, unitaries, training_data, _ = random_network(arch, samples)
        model = HybridGraphQCNNQML(arch, conv_layers)
        return model, training_data


__all__ = [
    "HybridGraphQCNNQML",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
