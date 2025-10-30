"""GraphQNNGen262 – Quantum‑neural‑network with quantum‑kernel support.

The quantum implementation mirrors the classical API but replaces linear
transformations with unitary evolution and incorporates a TorchQuantum
ansatz for kernel evaluation.  The class can be instantiated with a
``kernel_type`` of ``"quantum"`` to use a variational circuit for
similarity estimation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

Tensor = torch.Tensor

__all__ = [
    "GraphQNNGen262",
]


class GraphQNNGen262:
    """Hybrid quantum graph neural network with a quantum kernel.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the virtual quantum neural network.
    n_wires : int, default=4
        Number of qubits in the base device.
    kernel_type : str, default="quantum"
        Kernel type; ``"quantum"`` uses a TorchQuantum ansatz,
        ``"none"`` disables kernel‑based graph construction.
    """

    def __init__(self, qnn_arch: Sequence[int], n_wires: int = 4, kernel_type: str = "quantum") -> None:
        self.qnn_arch = list(qnn_arch)
        self.n_wires = n_wires
        self.kernel_type = kernel_type
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    # ------------------------------------------------------------------ #
    #  Data generation utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = GraphQNNGen262._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        """Generate a random quantum network and training data."""
        target_unitary = GraphQNNGen262._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNNGen262.random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = GraphQNNGen262._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(
                        GraphQNNGen262._random_qubit_unitary(num_inputs + 1),
                        GraphQNNGen262._tensored_id(num_outputs - 1),
                    )
                    op = GraphQNNGen262._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------ #
    #  Forward propagation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return GraphQNNGen262._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, GraphQNNGen262._tensored_zero(num_outputs))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return GraphQNNGen262._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    @staticmethod
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
                current_state = GraphQNNGen262._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    # ------------------------------------------------------------------ #
    #  Fidelity helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the absolute squared overlap between pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen262.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Quantum kernel utilities
    # ------------------------------------------------------------------ #
    class KernalAnsatz(tq.QuantumModule):
        """Variational ansatz that encodes two classical vectors as a quantum state."""

        def __init__(self, func_list: List[dict]) -> None:
            super().__init__()
            self.func_list = func_list

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
            q_device.reset_states(x.shape[0])
            for info in self.func_list:
                params = (
                    x[:, info["input_idx"]]
                    if tq.op_name_dict[info["func"]].num_params
                    else None
                )
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            for info in reversed(self.func_list):
                params = (
                    -y[:, info["input_idx"]]
                    if tq.op_name_dict[info["func"]].num_params
                    else None
                )
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    class Kernel(tq.QuantumModule):
        """Quantum kernel that evaluates the overlap of two encoded states."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = GraphQNNGen262.KernalAnsatz(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[qt.Qobj], b: Sequence[qt.Qobj]) -> np.ndarray:
        """Evaluate the Gram matrix between two sets of quantum states."""
        mat = np.empty((len(a), len(b)), dtype=float)
        for i, state_i in enumerate(a):
            for j, state_j in enumerate(b):
                mat[i, j] = GraphQNNGen262.state_fidelity(state_i, state_j)
        return mat

    def graph_from_kernel(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build an adjacency graph from a quantum kernel matrix."""
        mat = self.kernel_matrix(states, states)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                weight = mat[i, j]
                if weight >= threshold:
                    graph.add_edge(i, j, weight=weight)
                elif secondary is not None and weight >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
