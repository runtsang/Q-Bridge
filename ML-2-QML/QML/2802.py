"""Hybrid quantum graph neural network with convolutional encoder.

This module mirrors the classical ``GraphQNNHybrid`` but replaces the linear
stack with a quantum variational circuit that processes the encoded feature
vector.  The interface and helper functions are kept identical to the
original GraphQNN utilities so that data pipelines can be swapped
freely between the classical and quantum back‑ends.

The quantum encoder uses TorchQuantum's ``GeneralEncoder`` (a 4‑qubit
parameter‑free circuit) to map the CNN output into a quantum state.
Each subsequent layer applies a set of random unitaries followed by a
partial trace that reduces the number of qubits according to the chosen
architecture.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Core utilities – identical to the classical module but with Qobj objects
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> tq.Qobj:
    """Return the identity operator on ``num_qubits`` qubits."""
    identity = tq.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> tq.Qobj:
    """Return the projector onto |0⟩^⊗n."""
    projector = tq.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: tq.Qobj, source: int, target: int) -> tq.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> tq.Qobj:
    dim = 2 ** num_qubits
    matrix = torch.randn(dim, dim) + 1j * torch.randn(dim, dim)
    unitary = torch.linalg.orth(matrix)
    qobj = tq.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> tq.Qobj:
    dim = 2 ** num_qubits
    amplitudes = torch.randn(dim, 1) + 1j * torch.randn(dim, 1)
    amplitudes /= torch.linalg.norm(amplitudes)
    state = tq.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: tq.Qobj, samples: int) -> List[Tuple[tq.Qobj, tq.Qobj]]:
    """Generate input–output pairs for a target unitary."""
    dataset: List[Tuple[tq.Qobj, tq.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Construct a random quantum network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[tq.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[tq.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = tq.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: tq.Qobj, keep: Sequence[int]) -> tq.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: tq.Qobj, remove: Sequence[int]) -> tq.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[tq.Qobj]],
    layer: int,
    input_state: tq.Qobj,
) -> tq.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = tq.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[tq.Qobj]],
    samples: Iterable[Tuple[tq.Qobj, tq.Qobj]],
) -> List[List[tq.Qobj]]:
    """Run a forward pass through a quantum feed‑forward circuit."""
    stored_states: List[List[tq.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: tq.Qobj, b: tq.Qobj) -> float:
    """Absolute squared overlap between pure quantum states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[tq.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from state fidelities."""
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
#  Hybrid quantum model
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(tq.QuantumModule):
    """Quantum variant of the hybrid graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes for the variational circuit (e.g. ``[16, 32, 64]``).
    n_wires : int, optional
        Number of qubits used for the convolutional encoder.
    """

    def __init__(
        self,
        arch: Sequence[int],
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.n_wires = n_wires

        # Convolutional encoder (Quantum‑NAT style)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # Build a list of unitaries for each layer
        self.unitaries: List[List[tq.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            layer_ops: List[tq.Qobj] = []
            for _ in range(self.arch[layer]):
                op = _random_qubit_unitary(self.arch[layer - 1] + 1)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder + variational circuit."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode the CNN‑derived feature vector (average‑pooling to match 4‑wire encoder)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Apply each layer of random unitaries, followed by a partial trace
        for layer_idx, layer_ops in enumerate(self.unitaries[1:], start=1):
            for op in layer_ops:
                op(qdev)
            # Reduce the number of qubits to match the next layer
            keep = list(range(self.arch[layer_idx - 1]))
            qdev = qdev.ptrace(keep)

        out = self.measure(qdev)
        return self.norm(out)

    # --------------------------------------------------------------------- #
    #  Compatibility helpers – identical to the original GraphQNN utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(*args, **kwargs):
        return random_network(*args, **kwargs)

    @staticmethod
    def random_training_data(*args, **kwargs):
        return random_training_data(*args, **kwargs)

    @staticmethod
    def feedforward(*args, **kwargs):
        return feedforward(*args, **kwargs)

    @staticmethod
    def state_fidelity(*args, **kwargs):
        return state_fidelity(*args, **kwargs)

    @staticmethod
    def fidelity_adjacency(*args, **kwargs):
        return fidelity_adjacency(*args, **kwargs)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
