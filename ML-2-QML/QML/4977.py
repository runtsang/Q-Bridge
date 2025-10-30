"""Hybrid quantum implementation of QuantumNATGen222.

This module keeps the same public API as the classical counterpart
while replacing the CNN+FC pipeline with a variational quantum circuit.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

# --------------------------------------------------------------------------- #
#  Quantum layer definitions
# --------------------------------------------------------------------------- #

class _QuantumLayer(tq.QuantumModule):
    """Randomised variational block followed by a fixed gate pattern."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(
            n_ops=50, wires=list(range(self.n_wires))
        )
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


# --------------------------------------------------------------------------- #
#  Fast Estimator (quantum)
# --------------------------------------------------------------------------- #

class FastEstimator:
    """Expectation value evaluator for a parameterised QuantumCircuit with optional shot noise."""

    def __init__(self, circuit: tq.QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> tq.QuantumCircuit:
        if len(param_vals)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[tq.QuantumOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            circ = self._bind(vals)
            state = tq.StateVector(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        return [
            [rng.normal(r.real, max(1e-6, 1 / shots)) + 1j * rng.normal(r.imag, max(1e-6, 1 / shots))
             for r in row]
            for row in results
        ]


# --------------------------------------------------------------------------- #
#  Graph utilities (quantum)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> "qt.Qobj":
    import qutip as qt
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> "qt.Qobj":
    import qutip as qt
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: "qt.Qobj", source: int, target: int) -> "qt.Qobj":
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> "qt.Qobj":
    import scipy as sc
    import qutip as qt
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> "qt.Qobj":
    import scipy as sc
    import qutip as qt
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: "qt.Qobj", samples: int) -> List[Tuple["qt.Qobj", "qt.Qobj"]]:
    dataset: List[Tuple["qt.Qobj", "qt.Qobj"]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        st = _random_qubit_state(n)
        dataset.append((st, unitary * st))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    from qutip import Qobj
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_in = qnn_arch[layer - 1]
        num_out = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for out_idx in range(num_out):
            op = _random_qubit_unitary(num_in + 1)
            if num_out > 1:
                op = qt.tensor(_random_qubit_unitary(num_in + 1), _tensored_id(num_out - 1))
                op = _swap_registers(op, num_in, num_in + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: "qt.Qobj", keep: Sequence[int]) -> "qt.Qobj":
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: "qt.Qobj", remove: Sequence[int]) -> "qt.Qobj":
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence["qt.Qobj"]],
    layer: int,
    input_state: "qt.Qobj",
) -> "qt.Qobj":
    num_in = qnn_arch[layer - 1]
    num_out = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_out))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_in))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence["qt.Qobj"]],
    samples: Iterable[Tuple["qt.Qobj", "qt.Qobj"]],
) -> List[List["qt.Qobj"]]:
    stored: List[List["qt.Qobj"]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

def state_fidelity(a: "qt.Qobj", b: "qt.Qobj") -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence["qt.Qobj"],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> "nx.Graph":
    import networkx as nx
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
#  Hybrid quantum class
# --------------------------------------------------------------------------- #

class QuantumNATGen222(tq.QuantumModule):
    """
    Quantum counterpart of :class:`QuantumNATGen222` from the classical module.
    Provides a variational circuit, measurement, and the same graph utilities.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        graph_arch: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = num_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = _QuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        self._graph_arch = list(graph_arch) if graph_arch else None
        if self._graph_arch:
            self.graph_arch, self.graph_unitaries, self.graph_training, self._target_unitary = random_network(
                self._graph_arch, samples=100
            )
        self.estimator = FastEstimator(self.to_qc())

    def to_qc(self) -> tq.QuantumCircuit:
        """Expose the underlying circuit for estimation."""
        qc = tq.QuantumCircuit(self.n_wires)
        qc.add_module(self.encoder)
        qc.add_module(self.q_layer)
        qc.add_module(self.measure)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a 2‑D image, run the circuit, and return a batch of probabilities."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    # Estimator wrapper
    def evaluate(
        self,
        observables: Iterable[tq.QuantumOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        return self.estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

    # Graph utilities
    @property
    def graph_arch(self) -> Sequence[int] | None:
        return self._graph_arch

    @property
    def graph_unitaries(self) -> List[List["qt.Qobj"]] | None:
        return self.graph_unitaries if hasattr(self, "graph_unitaries") else None

    def graph_feedforward(
        self, samples: Iterable[Tuple["qt.Qobj", "qt.Qobj"]]
    ) -> List[List["qt.Qobj"]]:
        if not self.graph_unitaries:
            raise RuntimeError("Graph network not initialized.")
        return feedforward(self._graph_arch, self.graph_unitaries, samples)

    def graph_fidelity_adjacency(
        self, threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> "nx.Graph":
        if not self.graph_unitaries:
            raise RuntimeError("Graph network not initialized.")
        states = [s[0] for s in self.graph_training]
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = ["QuantumNATGen222"]
