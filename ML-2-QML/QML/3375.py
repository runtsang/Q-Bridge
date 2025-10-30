"""Quantum Graph Neural Network implementation.

The class mirrors the classical interface but operates on qutip quantum states
and uses a TorchQuantum ansatz for kernel evaluation.  The network is a stack
of random unitary layers.  Fidelity‑based adjacency graphs and a quantum
RBF‑style kernel are provided.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Random unitary and state helpers
# ----------------------------------------------------------------------
def _tensored_id(num_qubits: int) -> qt.Qobj:
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    P0 = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    P0.dims = [dims.copy(), dims.copy()]
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
    U = sc.linalg.orth(mat)
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    state = qt.Qobj(vec)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

# ----------------------------------------------------------------------
# Quantum kernel (TorchQuantum)
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Programmable list of quantum gates that encode two inputs."""

    def __init__(self, func_list: List[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed ansatz that implements a simple RBF‑style quantum kernel."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
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

# ----------------------------------------------------------------------
# Quantum GraphQNN
# ----------------------------------------------------------------------
class GraphQNN:
    """Quantum graph neural network with fidelity‑based graph construction
    and a quantum RBF‑style kernel.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes (number of qubits in each layer).
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.qnn_arch = list(qnn_arch)
        self.unitaries: List[List[qt.Qobj]] = []
        self.kernel = Kernel(n_wires=self.qnn_arch[-1])

    # ------------------------------------------------------------------
    # Network construction helpers
    # ------------------------------------------------------------------
    def random_network(self, samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Generate a random network and synthetic training data."""
        target_unitary = _random_qubit_unitary(self.qnn_arch[-1])
        training_data = [
            (_random_qubit_state(self.qnn_arch[-1]), target_unitary * _random_qubit_state(self.qnn_arch[-1]))
            for _ in range(samples)
        ]

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out_idx in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + out_idx)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        self.unitaries = unitaries
        return self.qnn_arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _layer_channel(self, layer: int, input_state: qt.Qobj) -> qt.Qobj:
        """Apply a single layer of the network to an input state."""
        num_inputs = self.qnn_arch[layer - 1]
        num_outputs = self.qnn_arch[layer]
        state = qt.tensor(input_state, _tensored_zero(num_outputs))
        layer_unitary = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return layer_unitary * state * layer_unitary.dag()

    def feedforward(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        """Propagate each sample through all layers."""
        all_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise: List[qt.Qobj] = [sample]
            current = sample
            for layer in range(1, len(self.qnn_arch)):
                current = self._layer_channel(layer, current)
                layerwise.append(current)
            all_states.append(layerwise)
        return all_states

    # ------------------------------------------------------------------
    # Fidelity‑based graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Absolute squared overlap of two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Quantum kernel utilities
    # ------------------------------------------------------------------
    def kernel_matrix(
        self,
        a: Sequence[qt.Qobj],
        b: Sequence[qt.Qobj]
    ) -> np.ndarray:
        """Compute the Gram matrix using the embedded quantum kernel."""
        mat = np.empty((len(a), len(b)))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                # Convert Qobj to torch tensor of amplitudes
                xi_t = torch.tensor(xi.full().flatten(), dtype=torch.float32)
                yj_t = torch.tensor(yj.full().flatten(), dtype=torch.float32)
                mat[i, j] = self.kernel.forward(xi_t, yj_t).item()
        return mat

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    "GraphQNN",
    "Kernel",
    "KernalAnsatz",
]
