"""Combined quantum graph‑based regression module.

This module mirrors the classical implementation above but uses
torchquantum to encode and process quantum states.  The same
:class:`GraphQNNRegression` name is reused for consistency.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import scipy as sc
import torch
import torch.nn as nn
import torchquantum as tq
from qutip import Qobj

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  QNN utilities (from original QNN.py)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> Qobj:
    identity = Qobj(np.eye(2 ** num_qubits))
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> Qobj:
    projector = Qobj(np.diag([1.0] + [0.0] * (2 ** num_qubits - 1)))
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: Qobj, samples: int) -> list[tuple[Qobj, Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = Qobj(np.kron(_random_qubit_unitary(num_inputs + 1), np.eye(2 ** (num_outputs - 1))))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Qobj]],
    layer: int,
    input_state: Qobj,
) -> Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = Qobj(np.kron(input_state.full(), np.eye(2 ** num_outputs)))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Qobj]],
    samples: Iterable[tuple[Qobj, Qobj]],
) -> List[List[Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[Qobj],
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
#  Quantum regression data (from QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Quantum regression model
# --------------------------------------------------------------------------- #

class QModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
#  GraphQNNRegression wrapper (quantum side)
# --------------------------------------------------------------------------- #

class GraphQNNRegression:
    """Quantum wrapper that builds a random QNN, a regression dataset,
    and a :class:`QModel` that learns to map quantum states to real
    targets.  The public API mirrors the classical counterpart.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_wires: int,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
        device: torch.device | str = "cpu",
    ):
        # Build random quantum network
        self.arch, self.unitaries, self.train_data, self.target_unitary = random_network(
            qnn_arch, samples=100
        )

        # Compute adjacency graph of the target unitary's layers
        self.graph = fidelity_adjacency(
            [u for layer in self.unitaries for u in layer], graph_threshold, secondary=secondary
        )

        # Build dataset
        self.dataset = RegressionDataset(samples=2000, num_wires=num_wires)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Build model
        self.model = QModel(num_wires).to(device)
        self.device = device

    def train(self, epochs: int = 10, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                states = batch["states"].to(self.device)
                targets = batch["target"].to(self.device)
                preds = self.model(states)
                loss = criterion(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(self.dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(states.to(self.device)).cpu()

__all__ = [
    "GraphQNNRegression",
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
