"""GraphQNNHybrid – quantum implementation.

All public methods mirror the classical version but operate on
QuTiP objects and Qiskit circuits.  The class can generate random
tensor‑product unitary networks, propagate quantum states,
compute fidelity graphs, and build a variational auto‑encoder circuit.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple, Union

import networkx as nx
import qutip as qt
import scipy as sc

Tensor = qt.Qobj


# --------------------------------------------------------------------------- #
# Random network helpers
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    I = qt.qeye(2 ** num_qubits)
    I.dims = [[2] * num_qubits, [2] * num_qubits]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    P0 = qt.fock(2 ** num_qubits).proj()
    P0.dims = [[2] * num_qubits, [2] * num_qubits]
    return P0


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    U, _ = sc.linalg.qr(mat)
    q = qt.Qobj(U)
    q.dims = [[2] * num_qubits, [2] * num_qubits]
    return q


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    s = qt.Qobj(vec)
    s.dims = [[2] * num_qubits, [1] * num_qubits]
    return s


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (state, U*state) pairs."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        s = _random_qubit_state(n)
        data.append((s, unitary * s))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return (arch, list_of_unitary_lists, training_data, target_unitary)."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    train_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_in, num_out = qnn_arch[layer - 1], qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out_idx in range(num_out):
            op = _random_qubit_unitary(num_in + 1)
            if num_out > 1:
                op = qt.tensor(_random_qubit_unitary(num_in + 1), _tensored_id(num_out - 1))
                op = _swap_registers(op, num_in, num_in + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, train_data, target_unitary


# --------------------------------------------------------------------------- #
# Forward propagation
# --------------------------------------------------------------------------- #
def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int],
                   unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    num_in, num_out = qnn_arch[layer - 1], qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_out))

    U = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        U = gate * U

    return _partial_trace_remove(U * state * U.dag(), range(num_in))


def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Return state trajectories for each sample."""
    trajectories: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        traj = [sample]
        cur = sample
        for layer in range(1, len(qnn_arch)):
            cur = _layer_channel(qnn_arch, unitaries, layer, cur)
            traj.append(cur)
        trajectories.append(traj)
    return trajectories


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return float(abs((a.dag() * b)[0, 0]) ** 2)


def fidelity_adjacency(states: Sequence[qt.Qobj],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, bj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder circuit (variational)
# --------------------------------------------------------------------------- #
def autoencoder(num_latent: int, num_trash: int):
    """Return a Qiskit circuit that implements a variational auto‑encoder."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.primitives import StatevectorSampler as Sampler

    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Feature embedding + ansatz
    embed = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(embed, range(num_latent + num_trash))

    qc.barrier()

    # Swap‑test for reconstruction
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Wrap in a SamplerQNN for use as a variational layer
    qnn = SamplerQNN(circuit=qc,
                     input_params=[],
                     weight_params=qc.parameters,
                     interpret=lambda x: x,
                     output_shape=2,
                     sampler=Sampler())
    return qnn


# --------------------------------------------------------------------------- #
# Unified hybrid class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Hybrid graph neural network that can operate on classical tensors or quantum states.

    Instantiating with `quantum=True` switches the backend; all public methods
    are identical, enabling seamless testing of different computational
    paradigms on the same graph data.
    """
    def __init__(self, quantum: bool = False):
        self.quantum = quantum

    # ------------------ Classical API ------------------
    def random_network(self, qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    def feedforward(self, qnn_arch: Sequence[int],
                    weights: Sequence[Tensor],
                    samples: Iterable[Tuple[Tensor, Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self, states: Sequence[Tensor],
                           threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ------------------ Quantum API ------------------
    def random_network_q(self, qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    def feedforward_q(self, qnn_arch: Sequence[int],
                      unitaries: Sequence[Sequence[qt.Qobj]],
                      samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
        return feedforward(qnn_arch, unitaries, samples)

    def state_fidelity_q(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency_q(self, states: Sequence[qt.Qobj],
                             threshold: float,
                             *, secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    def autoencoder_q(self, num_latent: int, num_trash: int) -> qt.Qobj:
        return autoencoder(num_latent, num_trash)

__all__ = [
    "GraphQNNHybrid",
    "autoencoder",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
