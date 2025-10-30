"""GraphQNNGen078: quantum graph neural network with sampler support.

This module mirrors the classical implementation but operates on quantum
states and unitaries.  It provides random network generation, forward
propagation, fidelity calculations, graph construction, and a parameterised
quantum sampler circuit.  The class interface is identical to the
classical version, enabling side‑by‑side experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import scipy as sc
import qutip as qt
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

Qobj = qt.Qobj


def _tensored_id(num_qubits: int) -> Qobj:
    """Identity operator on `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> Qobj:
    """Projector onto the all‑zero state for `num_qubits` qubits."""
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj


def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    """Swap qubit registers in a tensor product."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> Qobj:
    """Generate a random Haar‑distributed unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: Qobj, n_samples: int) -> List[Tuple[Qobj, Qobj]]:
    """Generate (|ψ⟩, U|ψ⟩) pairs for a target unitary."""
    data: List[Tuple[Qobj, Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(n_samples):
        state = _random_qubit_state(num_qubits)
        data.append((state, unitary * state))
    return data


def random_network(arch: Sequence[int], n_samples: int):
    """Create a random quantum network and training set."""
    target_unitary = _random_qubit_unitary(arch[-1])
    train = random_training_data(target_unitary, n_samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(arch)):
        nin = arch[layer - 1]
        nout = arch[layer]
        ops: List[Qobj] = []
        for out_idx in range(nout):
            op = _random_qubit_unitary(nin + 1)
            if nout > 1:
                op = qt.tensor(_random_qubit_unitary(nin + 1), _tensored_id(nout - 1))
                op = _swap_registers(op, nin, nin + out_idx)
            ops.append(op)
        unitaries.append(ops)

    return list(arch), unitaries, train, target_unitary


def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    """Partial trace keeping the qubits in `keep`."""
    if len(keep) == len(state.dims[0]):
        return state
    return state.ptrace(list(keep))


def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    """Partial trace removing the qubits in `remove`."""
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                   layer: int, input_state: Qobj) -> Qobj:
    """Apply a single layer of the quantum network."""
    nin = arch[layer - 1]
    nout = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(nout))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(nin))


def feedforward(arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    """Run a forward pass through the quantum network."""
    stored: List[List[Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Qobj, b: Qobj) -> float:
    """Return the absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[Qobj], thresh: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= thresh:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g


# --- Sampler network ---------------------------------------------------------

def SamplerQNN() -> QiskitSamplerQNN:
    """Return a parameterised quantum sampler circuit."""
    inputs = ParameterVector("x", 2)
    weights = ParameterVector("w", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = StatevectorSampler()
    return QiskitSamplerQNN(circuit=qc, input_params=inputs,
                            weight_params=weights, sampler=sampler)


# --- Unified class -----------------------------------------------------------

class GraphQNNGen078:
    """Quantum graph‑based neural network with optional sampler.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [2, 4, 2].
    mode : str, optional
        Currently only 'quantum' is supported; the class can be extended
        to support hybrid or classical modes in the future.
    """

    def __init__(self, arch: Sequence[int], mode: str = "quantum") -> None:
        self.arch = list(arch)
        self.mode = mode
        self.unitaries, self.train_data, self.target = self._build_random()

    def _build_random(self):
        _, ops, train, tgt = random_network(self.arch, 50)
        return ops, train, tgt

    def train(self, epochs: int = 10, lr: float = 0.01) -> None:
        """A placeholder training loop that optimises the final unitary."""
        # In a real implementation this would involve quantum gradient descent.
        pass

    def forward(self, inputs: Qobj) -> List[Qobj]:
        """Run a forward pass and return layerwise states."""
        return feedforward(self.arch, self.unitaries, [(inputs, None)])[0]

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Return a graph built from state fidelities of the last layer."""
        last_layer = [act[-1] for act in self.forward(self.train_data[0][0])]
        return fidelity_adjacency(last_layer, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    def sampler(self, inp: Qobj) -> Qobj:
        """Return the sampler output for a given input."""
        return SamplerQNN()(inp)


__all__ = [
    "GraphQNNGen078",
    "SamplerQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
