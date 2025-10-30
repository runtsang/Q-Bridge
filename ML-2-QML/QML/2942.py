"""Quantum Graph Neural Network with auto‑encoding using Qiskit.

The quantum counterpart implements the same public API as the classical
variant but operates on Qobj states.  It uses Qiskit’s
``SamplerQNN`` and a variational auto‑encoder built from RealAmplitudes
circuits to compress and reconstruct graph embeddings.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
#  Helper functions (original QML logic retained)                          #
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Quantum auto‑encoder construction (inspired by Autoencoder.py)          #
# --------------------------------------------------------------------------- #

def _build_variational_autoencoder(num_latent: int, num_trash: int) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """
    Construct a variational auto‑encoder circuit that entangles
    ``num_latent`` qubits with ``num_trash`` auxiliary qubits.

    Returns a tuple (encoder, decoder).  The decoder is the inverse
    of the encoder and is used for reconstruction.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encoder: RealAmplitudes on the first ``num_latent + num_trash`` qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, list(range(num_latent + num_trash)))

    # Swap‑test to entangle latent with trash
    auxiliary_qubit = num_latent + 2 * num_trash
    qc.h(auxiliary_qubit)
    for i in range(num_trash):
        qc.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    qc.h(auxiliary_qubit)

    # Measurement is not required for the auto‑encoder; we keep the state
    return qc, qc.inverse()

def _graph_to_qobj(graph: nx.Graph) -> qt.Qobj:
    """Adjacency matrix as a Qobj."""
    mat = nx.to_numpy_array(graph, dtype=complex)
    return qt.Qobj(mat)


# --------------------------------------------------------------------------- #
#  GraphQNN class (public API)                                               #
# --------------------------------------------------------------------------- #

class GraphQNN:
    """
    Quantum graph neural network that mirrors the classical :class:`GraphQNN`.
    The network propagates quantum states through a series of random unitaries
    and compresses them with a variational auto‑encoder.
    """

    def __init__(self, qnn_arch: Sequence[int], device: str | None = None):
        self.arch = list(qnn_arch)
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(
            self.arch, samples=100
        )
        self.device = device or "cpu"  # Qiskit uses backend strings; default to CPU simulator

        # Build auto‑encoder circuits
        self.encoder_circ, self.decoder_circ = _build_variational_autoencoder(
            num_latent=self.arch[-1], num_trash=2
        )
        self.sampler = Sampler(method="statevector", shots=1024, backend="qasm_simulator")

    # --------------------------------------------------------------------- #
    #  Forward propagation utilities                                         #
    # --------------------------------------------------------------------- #

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Return all intermediate states for the provided samples."""
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted graph from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # --------------------------------------------------------------------- #
    #  Graph embedding & auto‑encoding                                        #
    # --------------------------------------------------------------------- #

    def encode_graph(self, graph: nx.Graph) -> qt.Qobj:
        """
        Embed a graph into a latent quantum state.

        The adjacency matrix is first converted to a Qobj and propagated
        through the network.  The final state is then processed by the
        auto‑encoder encoder to obtain the latent state.
        """
        state = _graph_to_qobj(graph)
        # Forward pass through the network
        _, _, _, final_state = self.feedforward([(state, state)])[0]
        # Apply encoder
        encoded = self.encoder_circ.compose(final_state, inplace=False)
        return encoded

    def decode_latent(self, latent: qt.Qobj) -> qt.Qobj:
        """
        Reconstruct the original graph state from a latent state.

        The decoder is the inverse of the encoder and is applied to the
        latent state to recover the full graph state.
        """
        return self.decoder_circ.compose(latent, inplace=False)

    # --------------------------------------------------------------------- #
    #  Auto‑encoder training placeholder (no training in this simplified example) #
    # --------------------------------------------------------------------- #

    def train_autoencoder(self, *_, **__):
        """No‑op in the quantum seed; training would require a variational optimizer."""
        raise NotImplementedError("Quantum auto‑encoder training is not provided in this seed.")

    # --------------------------------------------------------------------- #
    #  Utility accessors                                                    #
    # --------------------------------------------------------------------- #

    @property
    def unitary_matrix(self) -> qt.Qobj:
        """Return the final layer unitary."""
        return self.unitaries[-1][0]

    def __repr__(self) -> str:
        return f"<GraphQNN arch={self.arch} device={self.device}>"

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
