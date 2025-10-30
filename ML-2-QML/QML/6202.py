"""
Graph‑Quantum Neural Network with a quantum autoencoder.
The quantum class mirrors GraphQNNAutoencoder, but replaces the
classical linear layers with variational unitaries and uses a
SamplerQNN‑based autoencoder to embed input states.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qutip as qt
import scipy as sc
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

Tensor = qt.Qobj


# --------------------------------------------------------------------------- #
# 1. Quantum autoencoder helper
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    iden = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    iden.dims = [dims.copy(), dims.copy()]
    return iden


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero


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


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    data = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n)
        data.append((state, unitary * state))
    return data


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        inputs = qnn_arch[layer - 1]
        outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for out_idx in range(outputs):
            op = _random_qubit_unitary(inputs + 1)
            if outputs > 1:
                op = qt.tensor(_random_qubit_unitary(inputs + 1), _tensored_id(outputs - 1))
                op = _swap_registers(op, inputs, inputs + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep) == len(state.dims[0]):
        return state
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
    num_in = qnn_arch[layer - 1]
    num_out = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_out))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_in))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
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
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 2. Quantum autoencoder circuit (SamplerQNN)
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz for the latent sub‑space
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test layer to entangle latent and trash qubits
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


def SamplerQNN_autoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    circuit = quantum_autoencoder_circuit(num_latent, num_trash)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# 3. Hybrid Graph‑QNN with quantum autoencoder
# --------------------------------------------------------------------------- #
class GraphQNNAutoencoder:
    """
    Quantum GNN that first encodes input states with a SamplerQNN‑based
    autoencoder and then propagates them through a stack of variational
    unitaries.  The API matches the classical counterpart.
    """

    def __init__(self, qnn_arch: Sequence[int], num_trash: int = 2):
        self.qnn_arch = list(qnn_arch)
        self.num_trash = num_trash
        # Build a quantum autoencoder for the first layer
        self.autoencoder = SamplerQNN_autoencoder(qnn_arch[0], num_trash)
        # Generate a random network of unitaries for the remaining layers
        _, self.unitaries, self.training_data, self.target_unitary = random_network(qnn_arch, samples=10)

    def encode_inputs(self, states: List[qt.Qobj]) -> List[qt.Qobj]:
        """Encode a batch of input states via the autoencoder."""
        return [self.autoencoder(state) for state in states]

    def forward(self, states: List[qt.Qobj]) -> List[qt.Qobj]:
        """Propagate encoded states through the variational layers."""
        encoded = self.encode_inputs(states)
        outputs = []
        for s in encoded:
            current = s
            for layer in range(1, len(self.qnn_arch)):
                current = _layer_channel(self.qnn_arch, self.unitaries, layer, current)
            outputs.append(current)
        return outputs

    def fidelity_graph(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)


__all__ = [
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "SamplerQNN_autoencoder",
    "GraphQNNAutoencoder",
]
