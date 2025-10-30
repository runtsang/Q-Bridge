"""Quantum implementation of Graph‑QNN with self‑attention and regression.

The module combines the QML‑GraphQNN, QML‑SelfAttention and QML‑Regression
seeds into a single quantum‑aware class `GraphQNNAttentionRegression`.  It
uses TorchQuantum for differentiable quantum circuits and Qiskit for
parameterised attention gates.  All components expose the same API as the
classical counterpart.

Usage
-----
>>> from GraphQNN__gen218 import GraphQNNAttentionRegression, RegressionDataset
>>> model = GraphQNNAttentionRegression([4, 8, 4])
>>> dataset = RegressionDataset(samples=200, num_wires=4)
>>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
>>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
>>> loss_fn = nn.MSELoss()
>>> for epoch in range(10):
...     for batch in loader:
...         loss = model.train_step(optimizer, loss_fn, batch)
...
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Quantum helpers (mirroring the original GraphQNN seed)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> tq.Qobj:
    return tq.eye(2 ** num_qubits)


def _tensored_zero(num_qubits: int) -> tq.Qobj:
    return tq.fock(2 ** num_qubits).proj()


def _swap_registers(op: tq.Qobj, source: int, target: int) -> tq.Qobj:
    order = list(range(op.dims[0].size))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> tq.Qobj:
    dim = 2 ** num_qubits
    mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    mat = np.linalg.svd(mat)[0]
    return tq.Qobj(mat)


def _random_qubit_state(num_qubits: int) -> tq.Qobj:
    dim = 2 ** num_qubits
    amp = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amp /= np.linalg.norm(amp)
    return tq.Qobj(amp)


def random_training_data(unitary: tq.Qobj, samples: int) -> list[tuple[tq.Qobj, tq.Qobj]]:
    dataset: list[tuple[tq.Qobj, tq.Qobj]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[tq.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[tq.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = tq.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: tq.Qobj, keep: Sequence[int]) -> tq.Qobj:
    return state.ptrace(list(keep))


def _partial_trace_remove(state: tq.Qobj, remove: Sequence[int]) -> tq.Qobj:
    keep = list(range(state.shape[0]))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[tq.Qobj]], layer: int, input_state: tq.Qobj) -> tq.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = tq.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary

    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[tq.Qobj]], samples: Iterable[tuple[tq.Qobj, tq.Qobj]]) -> list[list[tq.Qobj]]:
    stored_states: list[list[tq.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: tq.Qobj, b: tq.Qobj) -> float:
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[tq.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
#  Quantum self‑attention (mirroring the original SelfAttention seed)
# --------------------------------------------------------------------------- #

class QuantumSelfAttention(tq.QuantumModule):
    """Parameterised circuit that implements a dot‑product style self‑attention."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.n_qubits):
            self.rx(qdev, wires=i, params=self.rotation_params[i, 0])
            self.ry(qdev, wires=i, params=self.rotation_params[i, 1])
            self.rz(qdev, wires=i, params=self.rotation_params[i, 2])
        for i in range(self.n_qubits - 1):
            self.crx(qdev, wires=(i, i + 1), params=self.entangle_params[i])
        measure = tq.MeasureAll(tq.PauliZ)
        return measure(qdev)


# --------------------------------------------------------------------------- #
#  Quantum regression utilities (mirroring the original QuantumRegression seed)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning quantum state vectors and regression labels."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  GraphQNNAttentionRegression class (quantum implementation)
# --------------------------------------------------------------------------- #

class GraphQNNAttentionRegression(tq.QuantumModule):
    """
    Quantum Graph‑QNN + self‑attention + regression head.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Widths of each QNN layer.
    attention_n_qubits : int, default 4
        Number of qubits used by the self‑attention module.
    regression_wires : int | None, default None
        Number of wires fed to the regression head.  If ``None`` the
        number of attention qubits is used.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        attention_n_qubits: int = 4,
        regression_wires: int | None = None,
    ):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        _, self.unitaries, _, _ = random_network(self.qnn_arch, samples=1)
        self.attention = QuantumSelfAttention(attention_n_qubits)
        self.regression_wires = regression_wires or attention_n_qubits
        self.head = nn.Linear(self.regression_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum QNN, self‑attention circuit, and regression head.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of input state vectors of shape ``(batch, 2**num_wires)``.

        Returns
        -------
        torch.Tensor
            Predicted scalars of shape ``(batch,)``.
        """
        # 1. Quantum QNN feedforward
        qdev_qnn = tq.QuantumDevice(
            n_wires=self.qnn_arch[-1], bsz=state_batch.shape[0], device=state_batch.device
        )
        encoder_qnn = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.qnn_arch[-1]}xRy"]
        )
        encoder_qnn(qdev_qnn, state_batch)

        for layer_ops in self.unitaries[1:]:
            for op in layer_ops:
                op(qdev_qnn)

        # Retrieve full quantum state (complex amplitudes)
        qnn_features = qdev_qnn.get_state()

        # 2. Quantum self‑attention
        qdev_attn = tq.QuantumDevice(
            n_wires=self.attention.n_qubits,
            bsz=state_batch.shape[0],
            device=state_batch.device,
        )
        encoder_attn = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.attention.n_qubits}xRy"]
        )
        encoder_attn(qdev_attn, qnn_features)

        attn_output = self.attention(qdev_attn)

        # 3. Regression head
        preds = self.head(attn_output)
        return preds.squeeze(-1)

    # --------------------------------------------------------------------- #
    #  Training utilities
    # --------------------------------------------------------------------- #
    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> float:
        optimizer.zero_grad()
        preds = self.forward(batch["states"])
        loss = loss_fn(preds, batch["target"])
        loss.backward()
        optimizer.step()
        return loss.item()

    # --------------------------------------------------------------------- #
    #  Evaluation utilities
    # --------------------------------------------------------------------- #
    def evaluate(self, data_loader: Iterable[dict[str, torch.Tensor]]) -> float:
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in data_loader:
                preds = self.forward(batch["states"])
                total += ((preds - batch["target"]) ** 2).sum().item()
                count += batch["states"].size(0)
        return total / count


__all__ = [
    "GraphQNNAttentionRegression",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "generate_superposition_data",
    "RegressionDataset",
]
