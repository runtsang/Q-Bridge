"""SharedClassName – quantum implementation.

This module reproduces the classical behaviour above but replaces linear
operations with parameterised quantum gates, uses Qiskit state‑vector
simulation for fidelity, and offers a minimal quantum transformer
block.  The API mirrors the classical side, making side‑by‑side
experiments trivial.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter

Tensor = np.ndarray  # for type hints; actual data is numpy arrays / Statevector objects


# --------------------------------------------------------------------------- #
#  Graph‑based utilities
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Statevector]:
    """Generate a random layered quantum network and training data."""
    dim_target = 2 ** qnn_arch[-1]
    # target unitary: Haar‑random matrix
    U = np.linalg.qr(np.random.randn(dim_target, dim_target) + 1j * np.random.randn(dim_target, dim_target))[0]
    target_unitary = Statevector.from_instruction(QuantumCircuit(qnn_arch[-1]).unitary(U))

    # training data: random input states and corresponding outputs
    training_data = [(Statevector.random(dim_target), Statevector.random(dim_target).evolve(target_unitary)) for _ in range(samples)]

    # per‑layer unitaries: each layer is a list of small random gates on the
    # input qubits plus an extra ancilla that is later traced out.
    unitaries: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        layer_ops: List[QuantumCircuit] = []
        for _ in range(qnn_arch[layer]):
            n_qubits = qnn_arch[layer - 1] + 1  # one ancilla per output
            qc = QuantumCircuit(n_qubits)
            # random single‑qubit rotations on all qubits
            for q in range(n_qubits):
                qc.ry(np.random.rand() * 2 * np.pi, q)
                qc.rz(np.random.rand() * 2 * np.pi, q)
            # random entangling
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)  # cycle
            layer_ops.append(qc)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[List[QuantumCircuit]], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
    """Forward propagation through a layered quantum network."""
    outputs: List[List[Statevector]] = []
    for state, _ in samples:
        current = state
        layerwise = [state]
        for layer in range(1, len(qnn_arch)):
            # apply the first gate of the layer and trace out ancilla
            qc = unitaries[layer][0]
            current = current.evolve(qc)
            current = _partial_trace_keep(current, list(range(qnn_arch[layer - 1])))
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs


def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Keep only the qubits in `keep`."""
    return state.trace_out(list(set(range(state.num_qubits)) - set(keep)))


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap of two pure quantum states."""
    return abs((a.data.conj().T @ b.data) ** 2)


def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Weighted graph from quantum state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a parameterised Qiskit circuit for a list of parameter sets."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[Parameter], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""

    def evaluate(self, observables: Iterable[Parameter], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean.real, max(1e-6, 1 / shots))) for mean in row] for row in raw]
        return noisy


# --------------------------------------------------------------------------- #
#  Quantum transformer (minimal – only illustrates the idea)
# --------------------------------------------------------------------------- #
class QuantumTransformer(nn.Module):
    """A tiny quantum transformer block that uses a single ancilla per head."""

    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 8) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.attn_circuit = QuantumCircuit(n_qubits)
        for head in range(num_heads):
            q = head % n_qubits
            self.attn_circuit.ry(np.random.rand() * 2 * np.pi, q)
        self.measure = self.attn_circuit.measure_all()

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover – placeholder
        """Placeholder to show API compatibility."""
        return x


# --------------------------------------------------------------------------- #
#  Quanvolution quantum filter
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(qiskit.circuit.QuantumCircuit):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self, name: str = "quanvolution_filter"):
        super().__init__(4, name=name)
        # Encode 4‑dimensional pixel vector into qubits
        for i in range(4):
            self.ry(np.random.rand() * 2 * np.pi, i)
        # Random entangling layer
        self.cx(0, 1)
        self.cx(2, 3)
        self.cx(0, 2)
        self.measure_all()

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover – placeholder
        return x


# --------------------------------------------------------------------------- #
#  SharedClassName – public façade
# --------------------------------------------------------------------------- #
class SharedClassName:
    """Facade that mirrors the classical API while delegating to quantum back‑ends."""

    # graph utilities
    random_network = staticmethod(random_network)
    feedforward = staticmethod(feedforward)
    state_fidelity = staticmethod(state_fidelity)
    fidelity_adjacency = staticmethod(fidelity_adjacency)

    # estimator classes
    FastBaseEstimator = FastBaseEstimator
    FastEstimator = FastEstimator

    # transformer
    QuantumTransformer = QuantumTransformer

    # quanvolution
    QuantumQuanvolutionFilter = QuantumQuanvolutionFilter


__all__ = [
    "SharedClassName",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastBaseEstimator",
    "FastEstimator",
    "QuantumTransformer",
    "QuantumQuanvolutionFilter",
]
