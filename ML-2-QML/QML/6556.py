"""Quantum‑kernel and graph construction using Qiskit.

The module contains:
- :class:`QKernel` – evaluates a quantum kernel via a parameter‑free circuit
  that encodes two classical vectors and computes the overlap.
- :class:`QGraph` – builds a weighted graph from state‑vector fidelities.
- :class:`QNN` – a depth‑controlled quantum neural network that can be
  instantiated with random unitaries per layer and optionally a classical
  encoder.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import itertools
import networkx as nx
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# 1. Quantum kernel utilities
# --------------------------------------------------------------------------- #
class QKernel:
    """Parameter‑free quantum kernel based on Ry rotations.

    The circuit encodes vector ``x`` into Ry gates, applies the same
    circuit for ``y`` with negative angles, and returns the absolute
    overlap of the resulting statevectors.
    """
    def __init__(self, n_wires: int | None = None) -> None:
        self.n_wires = n_wires
        self.backend = Aer.get_backend("statevector_simulator")

    def _build_circuit(self, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(x):
            qc.ry(val, i)
        for i, val in enumerate(y):
            qc.ry(-val, i)
        return qc

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        qc = self._build_circuit(x, y)
        result = execute(qc, self.backend).result()
        statevec = result.get_statevector(qc)
        return abs(statevec[0]) ** 2

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([[self(x, y) for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 2. Quantum graph utilities
# --------------------------------------------------------------------------- #
def qstate_fidelity(a: qi.Statevector, b: qi.Statevector) -> float:
    """Squared overlap of two statevectors."""
    return abs(np.vdot(a.data, b.data)) ** 2


def qfidelity_adjacency(
    states: Sequence[qi.Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = qstate_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 3. Random network generation (classical encoder + quantum layers)
# --------------------------------------------------------------------------- #
def random_training_data_qc(unitary: qi.Unitary, samples: int) -> List[Tuple[qi.Statevector, qi.Statevector]]:
    dataset: List[Tuple[qi.Statevector, qi.Statevector]] = []
    for _ in range(samples):
        # random pure state
        raw = np.random.randn(unitary.dim[0]) + 1j * np.random.randn(unitary.dim[0])
        raw /= np.linalg.norm(raw)
        state = qi.Statevector(raw)
        dataset.append((state, unitary @ state))
    return dataset


def random_network_qc(
    qnn_arch: List[int],
    samples: int,
    encoder: bool = False,
) -> Tuple[List[int], List[List[qi.Unitary]], List[Tuple[qi.Statevector, qi.Statevector]], qi.Unitary]:
    """Return architecture, list of layer unitaries, training data and target unitary."""
    target_unitary = qi.RandomUnitary(qnn_arch[-1]).data
    training_data = random_training_data_qc(target_unitary, samples)

    unitaries: List[List[qi.Unitary]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qi.Unitary] = []
        for output in range(num_outputs):
            op = qi.RandomUnitary(num_inputs + 1).data
            if num_outputs > 1:
                op = qi.tensor(qi.RandomUnitary(num_inputs + 1).data, qi.Identity(num_outputs - 1)).data
                # swap registers to match classical layout
                op = qi.swap(num_inputs, num_inputs + output).data @ op
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel_qc(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qi.Unitary]],
    layer: int,
    input_state: qi.Statevector,
) -> qi.Statevector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qi.tensor(input_state, qi.Identity(num_outputs)).data
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    full_state = layer_unitary @ state
    # trace out the newly added qubits
    keep = list(range(num_inputs))
    return qi.partial_trace(full_state, keep)


def feedforward_qc(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qi.Unitary]],
    samples: Iterable[Tuple[qi.Statevector, qi.Statevector]],
) -> List[List[qi.Statevector]]:
    """Return all layerwise states for each sample."""
    stored_states: List[List[qi.Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel_qc(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# 4. Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "QKernel",
    "QGraph",
    "qfidelity_adjacency",
    "random_network_qc",
    "feedforward_qc",
    "random_training_data_qc",
    "qstate_fidelity",
]
