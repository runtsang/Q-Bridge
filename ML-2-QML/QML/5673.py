"""Quantum implementation of graph neural networks that mirrors the classical API.

The module preserves all public symbols from the original GraphQNN module but
replaces linear layers with parameterised unitary blocks.  It also implements
the RBF kernel via a TorchQuantum ansatz, enabling a direct comparison between
classical and quantum kernels.

Key features:
* ``GraphQNNHybrid`` implements a variational circuit that matches the
  layer widths in ``architecture``.  Each layer consists of a random unitary
  acting on the input qubits plus an ancillary zero state.
* The forward pass returns the reduced state of the output qubits after
  each layer.
* The class exposes a ``quantum_kernel_matrix`` method that evaluates the
  overlap of two input states using the same ansatz as the classical RBF
  kernel, thus providing a seamless back‑to‑back substitution.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

Tensor = torch.Tensor
QObj = qt.Qobj

# --------------------------------------------------------------------
# Helper functions for qubit operators
# --------------------------------------------------------------------
def _tensored_identity(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    ident = qt.qeye(dim)
    dims = [2] * num_qubits
    ident.dims = [dims.copy(), dims.copy()]
    return ident

def _tensored_zero(num_qubits: int) -> QObj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero

def _swap_registers(op: QObj, src: int, tgt: int) -> QObj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    state = qt.Qobj(vec)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

# --------------------------------------------------------------------
# Core quantum graph neural network
# --------------------------------------------------------------------
class GraphQNNHybrid(tq.QuantumModule):
    """Quantum graph neural network with a variational ansatz.

    Parameters
    ----------
    architecture:
        Sequence of layer sizes.  The last element defines the number of output
        qubits.  Each layer receives the previous layer’s qubits plus a
        freshly prepared zero ancilla.
    n_wires:
        Total number of qubits used in the device.  Must be at least the largest
        layer size.
    """

    def __init__(self, architecture: Sequence[int], n_wires: int | None = None) -> None:
        super().__init__()
        self.arch = tuple(architecture)
        self.n_wires = max(architecture) if n_wires is None else n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.unitaries: List[List[QObj]] = [[]]  # first layer is identity
        # Build a random unitary ladder
        for layer in range(1, len(self.arch)):
            num_in = self.arch[layer - 1]
            num_out = self.arch[layer]
            ops: List[QObj] = []
            for out_idx in range(num_out):
                # each output qubit gets its own unitary on (in+out) qubits
                op = _random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(_random_qubit_unitary(num_in + 1), _tensored_identity(num_out - 1))
                    op = _swap_registers(op, num_in, num_in + out_idx)
                ops.append(op)
            self.unitaries.append(ops)

    # --------------------------------------------------------------------
    # Data generation
    # --------------------------------------------------------------------
    def random_training_data(self, unitary: QObj, samples: int) -> List[Tuple[QObj, QObj]]:
        """Generate input‑output pairs for a target unitary."""
        dataset = []
        n = len(unitary.dims[0])
        for _ in range(samples):
            state = _random_qubit_state(n)
            dataset.append((state, unitary * state))
        return dataset

    def random_network(self, samples: int) -> Tuple[Sequence[int], List[List[QObj]], List[Tuple[QObj, QObj]], QObj]:
        """Convenience wrapper that creates a random network and a training set."""
        target = _random_qubit_unitary(self.arch[-1])
        training = self.random_training_data(target, samples)
        return self.arch, self.unitaries, training, target

    # --------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------
    def _layer_channel(self, layer: int, input_state: QObj) -> QObj:
        """Apply the unitary block of a single layer and trace out the input qubits."""
        num_in = self.arch[layer - 1]
        num_out = self.arch[layer]
        # prepare ancilla
        state = qt.tensor(input_state, _tensored_zero(num_out))
        # compose all sub‑gates
        U = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            U = gate * U
        # propagate and trace out input qubits
        out_state = U * state * U.dag()
        return out_state.ptrace(range(num_in))

    def feedforward(
        self,
        unitaries: Sequence[Sequence[QObj]],
        samples: Sequence[Tuple[QObj, QObj]],
    ) -> List[List[QObj]]:
        """Return the state after each layer for every input sample."""
        all_states = []
        for inp, _ in samples:
            layer_states = [inp]
            current = inp
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, current)
                layer_states.append(current)
            all_states.append(layer_states)
        return all_states

    # --------------------------------------------------------------------
    # Fidelity utilities
    # --------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: QObj, b: QObj) -> float:
        """Squared overlap of two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[QObj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # --------------------------------------------------------------------
    # Quantum kernel (TorchQuantum implementation)
    # --------------------------------------------------------------------
    class KernalAnsatz(tq.QuantumModule):
        """Encodes two classical vectors into a single device state."""

        def __init__(self, func_list: List[dict]) -> None:
            super().__init__()
            self.func_list = func_list

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
            q_device.reset_states(x.shape[0])
            for info in self.func_list:
                params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            for info in reversed(self.func_list):
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    class Kernel(tq.QuantumModule):
        """Quantum kernel based on a fixed TorchQuantum ansatz."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = GraphQNNHybrid.KernalAnsatz(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

    def quantum_kernel_matrix(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Compute the quantum kernel Gram matrix using TorchQuantum."""
        kernel = self.Kernel()
        return np.array([[kernel(x, y).item() for y in b] for x in a])
