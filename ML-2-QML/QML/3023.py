"""Hybrid quantum estimator that fuses Qiskit circuits with quantum graph‑neural networks."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import qiskit
import qutip as qt
import scipy as sc

# --- Quantum utilities -----------------------------------------------------

def _tensored_identity(num_qubits: int) -> qt.Qobj:
    iden = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    iden.dims = [dims.copy(), dims.copy()]
    return iden

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero

def _swap_qubits(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    m = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    U = sc.linalg.orth(m)
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data_q(
    unitary: qt.Qobj,
    samples: int,
) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (input, target) pairs for training a quantum network."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        inp = _random_qubit_state(n)
        data.append((inp, unitary * inp))
    return data

def random_network_q(
    qnn_arch: List[int],
    samples: int,
) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Build a toy quantum‑graph‑NN with random unitaries."""
    target = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data_q(target, samples)
    layers: List[List[qt.Qobj]] = [[]]
    for layer_idx in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer_idx - 1]
        out_q = qnn_arch[layer_idx]
        ops: List[qt.Qobj] = []
        for out_idx in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            if out_q > 1:
                op = qt.tensor(_random_qubit_unitary(in_q + 1), _tensored_identity(out_q - 1))
                op = _swap_qubits(op, in_q, in_q + out_idx)
            ops.append(op)
        layers.append(ops)
    return qnn_arch, layers, training, target

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    return state.ptrace(list(keep))

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a layer’s unitaries and discard auxiliary qubits."""
    num_in = qnn_arch[layer - 1]
    num_out = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_out))
    op = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        op = gate * op
    out_state = op * state * op.dag()
    return _partial_trace_remove(out_state, range(num_in))

def feedforward_q(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Return layer‑wise states for each sample."""
    all_states: List[List[qt.Qobj]] = []
    for inp, _ in samples:
        states: List[qt.Qobj] = [inp]
        cur = inp
        for layer in range(1, len(qnn_arch)):
            cur = _layer_channel(qnn_arch, unitaries, layer, cur)
            states.append(cur)
        all_states.append(states)
    return all_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency_q(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build weighted graph from fidelities of quantum states."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --- Estimator implementation -----------------------------------------------

class FastBaseEstimator:
    """Hybrid quantum estimator that evaluates a Qiskit circuit or a quantum graph‑NN."""
    def __init__(
        self,
        circuit: qiskit.circuit.QuantumCircuit | None = None,
        qnn_arch: Sequence[int] | None = None,
        unitaries: List[List[qt.Qobj]] | None = None,
    ) -> None:
        self.circuit = circuit
        self.qnn_arch = qnn_arch
        self.unitaries = unitaries
        self._graph: nx.Graph | None = None

    # --------------------- Circuit evaluation --------------------------------

    def evaluate(
        self,
        observables: Iterable[qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for a parametrised Qiskit circuit."""
        if self.circuit is None:
            raise RuntimeError("No Qiskit circuit supplied.")
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind_params(params)
            state = qiskit.quantum_info.Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind_params(self, values: Sequence[float]) -> qiskit.circuit.QuantumCircuit:
        if len(values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.circuit.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate_with_shots(
        self,
        observables: Iterable[qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add shot‑noise to the deterministic expectation values."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy.append([rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row])
        return noisy

    # --------------------- Quantum graph‑NN evaluation -----------------------

    def feedforward_q(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Propagate a dataset through the stored quantum graph‑NN."""
        if self.qnn_arch is None or self.unitaries is None:
            raise RuntimeError("Quantum graph‑NN not configured.")
        return feedforward_q(self.qnn_arch, self.unitaries, samples)

    def build_graph_q(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted adjacency graph from quantum state fidelities."""
        self._graph = fidelity_adjacency_q(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
        return self._graph

    # --------------------- Convenience constructors --------------------------

    @classmethod
    def from_random_network_q(
        cls,
        qnn_arch: Sequence[int],
        samples: int,
    ) -> "FastBaseEstimator":
        arch, unitaries, training, _ = random_network_q(qnn_arch, samples)
        return cls(circuit=None, qnn_arch=arch, unitaries=unitaries)

    @classmethod
    def from_circuit(
        cls,
        circuit: qiskit.circuit.QuantumCircuit,
    ) -> "FastBaseEstimator":
        return cls(circuit=circuit)

__all__ = ["FastBaseEstimator"]
