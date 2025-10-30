"""GraphQNNGen207: quantum‑graph neural network.

The module mirrors the classical GraphQNNGen207 but replaces the
linear layers with parameterised quantum circuits that respect the
graph topology.  Fidelity‑based adjacency is computed from the outputs
of the last quantum layer, allowing a direct comparison between the
classical and quantum variants.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility helpers – quantum equivalents of the seed functions
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    identity = qt.qeye(dim)
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
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Return a random graph‑based quantum circuit together with data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Build a list of lists of unitaries – one list per layer
    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        nin = qnn_arch[layer - 1]
        nout = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out_idx in range(nout):
            op = _random_qubit_unitary(nin + 1)
            if nout > 1:
                op = qt.tensor(_random_qubit_unitary(nin + 1), _tensored_id(nout - 1))
                op = _swap_registers(op, nin, nin + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    nin = qnn_arch[layer - 1]
    nout = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(nout))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(nin))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Return the state at each layer for every sample."""
    outputs: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of pure states."""
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
# GraphQNNGen207 – quantum variant
# --------------------------------------------------------------------------- #

class GraphQNNGen207(tq.QuantumModule):
    """Quantum graph neural network that mirrors the classical
    GraphQNNGen207.  The network encodes a 2‑D image into a set of
    qubits, applies a stack of parameterised quantum layers that
    respect the graph topology, and measures all qubits in the
    Pauli‑Z basis.

    Parameters
    ----------
    conv_channels : Sequence[int]
        Number of channels for the convolutional encoder.
    qnn_arch : Sequence[int]
        Architecture of the graph‑based quantum circuit.
    n_wires : int, optional
        Number of qubits in the quantum device (default is the last
        layer of ``qnn_arch``).
    """

    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer that mimics the classical linear
        transformations in the graph network."""

        def __init__(self, nin: int, nout: int):
            super().__init__()
            self.nin = nin
            self.nout = nout
            # A small random layer – can be replaced with any
            # parameterised circuit from torchquantum
            self.random_layer = tq.RandomLayer(
                n_ops=10, wires=list(range(nin + nout))
            )
            # Single‑qubit rotations – trainable
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=self.nout - 1)
            self.crx(qdev, wires=[0, self.nout - 1])

    def __init__(
        self,
        conv_channels: Sequence[int] = (8, 16),
        qnn_arch: Sequence[int] = (64, 32, 16),
        n_wires: int | None = None,
    ) -> None:
        super().__init__()
        n_wires = n_wires or qnn_arch[-1]
        self.n_wires = n_wires

        # Encoder – identical to the QFCModel encoder
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )

        # Build a list of QLayer instances that match the graph
        self.layers: List[tq.QuantumModule] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.layers.append(self.QLayer(in_f, out_f))

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass on a batch of images.

        The image is first encoded into a quantum state, then passed
        through the stack of QLayers.  The final measurement is
        normalised with a classical BatchNorm.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )

        # Global average‑pool the image to a feature vector
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        self.encoder(qdev, pooled)

        for layer in self.layers:
            layer(qdev)

        out = self.measure(qdev)
        return self.norm(out)

    # ------------------------------------------------------------------ #
    # Compatibility helpers – expose the same API as the classical seed
    # ------------------------------------------------------------------ #
    def random_network(self, samples: int = 100):
        return random_network(self.qnn_arch, samples)

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
        return feedforward(self.qnn_arch, [l.random_layer for l in self.layers], samples)

    def fidelity_adjacency(self, states: Sequence[qt.Qobj], threshold: float):
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQNNGen207",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
