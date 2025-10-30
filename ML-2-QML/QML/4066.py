"""
HybridAutoencoder – Graph‑based quantum autoencoder built from Qutip unitaries.
It exposes the same ``encode``, ``decode`` and ``forward`` methods as the
classical implementation, allowing seamless substitution in a hybrid
pipeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
#  Utility helpers – adapted from the GraphQNN reference
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity on ``num_qubits`` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    identity.dims = [[2] * num_qubits, [2] * num_qubits]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector on ``num_qubits`` qubits."""
    proj = qt.fock(2 ** num_qubits).proj()
    proj.dims = [[2] * num_qubits, [2] * num_qubits]
    return proj


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate input‑output pairs for training a quantum autoencoder."""
    data = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        inp = _random_qubit_state(n)
        out = unitary * inp
        data.append((inp, out))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random graph‑based QNN and training data."""
    target = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f, out_f = qnn_arch[layer - 1], qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out_idx in range(out_f):
            op = _random_qubit_unitary(in_f + 1)
            if out_f > 1:
                op = qt.tensor(_random_qubit_unitary(in_f + 1), _tensored_id(out_f - 1))
                op = _swap_registers(op, in_f, in_f + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training, target


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, state: qt.Qobj) -> qt.Qobj:
    in_f, out_f = qnn_arch[layer - 1], qnn_arch[layer]
    state = qt.tensor(state, _tensored_zero(out_f))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(in_f))


def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Propagate a batch of states through the QNN."""
    outputs: List[List[qt.Qobj]] = []
    for inp, _ in samples:
        layerwise = [inp]
        cur = inp
        for layer in range(1, len(qnn_arch)):
            cur = _layer_channel(qnn_arch, unitaries, layer, cur)
            layerwise.append(cur)
        outputs.append(layerwise)
    return outputs


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g


# --------------------------------------------------------------------------- #
#  HybridAutoencoder – Quantum version
# --------------------------------------------------------------------------- #

class HybridAutoencoder:
    """
    Graph‑based quantum autoencoder.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes; the last element is the number of latent qubits.
    samples : int
        Number of training samples used to initialise random weights.

    Methods
    -------
    encode(state : qt.Qobj) -> qt.Qobj
        Propagate ``state`` through the encoder part of the QNN and return
        the latent state (last qubit(s)).
    decode(latent : qt.Qobj) -> qt.Qobj
        Reconstruct the full state from the latent using the decoder part.
    forward(state : qt.Qobj) -> qt.Qobj
        Full autoencoding: encode then decode.
    """

    def __init__(self, qnn_arch: Sequence[int], samples: int = 50):
        self.arch, self.unitaries, _, self.target_unitary = random_network(qnn_arch, samples)

    def encode(self, state: qt.Qobj) -> qt.Qobj:
        """Encode the input state to latent qubits."""
        cur = state
        for layer in range(1, len(self.arch) - 1):
            cur = _layer_channel(self.arch, self.unitaries, layer, cur)
        # Latent qubits are the last element of the architecture
        return cur

    def decode(self, latent: qt.Qobj) -> qt.Qobj:
        """Decode the latent back to full state."""
        cur = latent
        for layer in reversed(range(1, len(self.arch) - 1)):
            # Inverse of the encoder layer: apply the adjoint of the unitary
            inv_unitary = self.unitaries[layer][0].dag()
            cur = _partial_trace_keep(inv_unitary * cur * inv_unitary.dag(), list(range(len(cur.dims[0]))))
        return cur

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        """Full autoencoding: encode then decode."""
        latent = self.encode(state)
        return self.decode(latent)


__all__ = ["HybridAutoencoder"]
