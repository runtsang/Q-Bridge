"""Quantum‑graph neural network with integrated kernel and regression utilities.

The quantum counterpart mirrors the classical implementation while
leveraging TorchQuantum for variational layers and quantum state
encoding.  The public API is identical to the classical module,
enabling side‑by‑side experiments.

Highlights
----------
- Uses a Ry‑based data encoder and a configurable RandomLayer.
- Provides a quantum kernel via a fixed variational ansatz.
- Generates superposition training data.
- Builds fidelity‑based adjacency graphs from pure states.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Optional

import torch
import numpy as np
import torchquantum as tq
import qutip as qt
import networkx as nx
from torch.utils.data import Dataset
import torch.nn as nn
from torchquantum.functional import func_name_dict, op_name_dict

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Quantum Graph Neural Network
# --------------------------------------------------------------------------- #
class GraphQNNGen313(tq.QuantumModule):
    """Quantum‑graph neural network with a fixed encoder and trainable layer."""

    def __init__(self, arch: Sequence[int], gamma: float = 1.0):
        super().__init__()
        self.arch = list(arch)
        self.n_wires = self.arch[-1]  # assume last layer size equals number of qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"])
        self.q_layer = self._QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 1)
        self.kernel = _QuantumKernel(gamma)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, state_batch: Tensor) -> Tensor:
        """Map a batch of pure states to scalar predictions.

        Parameters
        ----------
        state_batch : torch.Tensor
            Complex states of shape ``(bsz, 2**n_wires)``.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        feats = self.measure(qdev)
        return self.head(feats).squeeze(-1)

    # --------------------------------------------------------------------- #
    #  Inner layer definition
    # --------------------------------------------------------------------- #
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    # --------------------------------------------------------------------- #
    #  Utility functions
    # --------------------------------------------------------------------- #
    @staticmethod
    def feedforward(
        arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        """Return state trajectories for each sample."""
        traj: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(arch)):
                current = _layer_channel(arch, unitaries, layer, current)
                layerwise.append(current)
            traj.append(layerwise)
        return traj

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = _state_fidelity_q(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random target unitary and training data."""
        target = _random_unitary(arch[-1])
        train_data = _random_training_data(target, samples)

        # Build a list of layers of random unitaries
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            in_n = arch[layer - 1]
            out_n = arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out_idx in range(out_n):
                op = _random_unitary(in_n + 1)
                if out_n > 1:
                    op = qt.tensor(_random_unitary(in_n + 1), _tensored_id(out_n - 1))
                    op = _swap_registers(op, in_n, in_n + out_idx)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return list(arch), unitaries, train_data, target

    # --------------------------------------------------------------------- #
    #  Kernel matrix
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[qt.Qobj], b: Sequence[qt.Qobj]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    # --------------------------------------------------------------------- #
    #  Regression dataset utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return state vectors and labels for the superposition regression task."""
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    return GraphQNNGen313.generate_superposition_data(num_wires, samples)

class RegressionDataset(Dataset):
    """Dataset for quantum superposition regression."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # noqa: D105
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # noqa: D105
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def _random_unitary(n: int) -> qt.Qobj:
    """Sample a random Haar‑distributed unitary."""
    dim = 2 ** n
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    matrix = np.linalg.qr(matrix)[0]  # orthonormal columns
    return qt.Qobj(matrix, dims=[[2] * n, [2] * n])

def _tensored_id(n: int) -> qt.Qobj:
    return qt.qeye(2 ** n, dims=[[2] * n, [2] * n])

def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)

def _random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    states = []
    for _ in range(samples):
        st = _random_state(unitary.dims[0].count(2))
        states.append((st, unitary * st))
    return states

def _random_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amp /= np.linalg.norm(amp)
    return qt.Qobj(amp, dims=[[2] * num_qubits, [1] * num_qubits])

def _state_fidelity_q(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def _layer_channel(arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, state: qt.Qobj) -> qt.Qobj:
    num_in = arch[layer - 1]
    num_out = arch[layer]
    state = qt.tensor(state, _tensored_id(num_out))
    # apply first unitary then successive ones
    unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        unitary = gate * unitary
    return _partial_trace_remove(unitary * state * unitary.dag(), range(num_in))

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

# --------------------------------------------------------------------------- #
#  Quantum kernel
# --------------------------------------------------------------------------- #
class _QuantumKernel(tq.QuantumModule):
    """Fixed Ansatz for a quantum kernel evaluation."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
        q_device.reset_states(x.shape[0])
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        return torch.abs(q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNNGen313",
    "RegressionDataset",
    "generate_superposition_data",
]
