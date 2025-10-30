"""Quantum version of **GraphQNN__gen092**.

This module keeps the original seed functions unchanged while adding
a new ``HybridGraphQNNQML`` class that implements a variational
graph‑based quantum neural network using Pennylane.  The class
provides a forward pass, a training loop with autograd, and
optional graph regularization.  The interface mirrors the classical
implementation so that existing scripts can swap between backends.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Callable

import networkx as nx
import numpy as np
import pennylane as qml
import torch
import qutip as qt

Tensor = torch.Tensor


# -------------------------------------------------------------
# Original seed utilities (unchanged)
# -------------------------------------------------------------
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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
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


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
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
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# -------------------------------------------------------------
# Helper functions for the QML class
# -------------------------------------------------------------
def _qobj_to_np(qobj: qt.Qobj) -> np.ndarray:
    """Convert a Qobj to a flat NumPy array."""
    return qobj.full().flatten()


def _np_to_qobj(vec: np.ndarray, dims: List[int]) -> qt.Qobj:
    """Convert a flat NumPy array to a Qobj with given dimensions."""
    return qt.Qobj(vec.reshape(dims), dims=[dims, [1] * len(dims)])


def _state_vector_tensor(state: qt.Qobj) -> Tensor:
    """Return a torch tensor of the flattened state vector."""
    return torch.tensor(state.full().flatten(), dtype=torch.float32)


def graph_regularizer(
    state_vectors: List[Tensor], graph: nx.Graph, weight: float = 1.0
) -> Tensor:
    """Compute a simple graph‑based regularization term for quantum states."""
    reg = torch.tensor(0.0, device=state_vectors[0].device)
    for i, j in graph.edges():
        reg += weight * torch.norm(state_vectors[i] - state_vectors[j]) ** 2
    return reg


# -------------------------------------------------------------
# Hybrid quantum graph neural network
# -------------------------------------------------------------
class HybridGraphQNNQML:
    """Variational graph‑based quantum neural network using Pennylane.

    The class mirrors the interface of the classical HybridGraphQNN
    but uses a Pennylane quantum device to implement each layer
    as a parameterised unitary on (in + 1) qubits.  A training loop
    with autograd is provided, and an optional graph regularizer can
    be applied to intermediate states.
    """

    def __init__(
        self,
        arch: Sequence[int],
        dev: qml.Device | None = None,
        seed: int | None = None,
    ):
        self.arch = list(arch)
        self.num_wires = max(self.arch)
        self.dev = dev or qml.device("default.qubit", wires=self.num_wires)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.params: dict[int, Tensor] = {}
        for layer in range(1, len(self.arch)):
            in_f = self.arch[layer - 1]
            out_f = self.arch[layer]
            # Parameters for a unitary on (in_f + 1) qubits
            num_params = 3 * (in_f + 1)  # 3 rotations per qubit
            self.params[layer] = torch.randn(out_f, num_params, requires_grad=True)

    # ---------------------------------------------------------
    # Circuit definition
    # ---------------------------------------------------------
    def _apply_layer_qml(
        self, state: qt.Qobj, layer: int, output_idx: int
    ) -> qt.Qobj:
        """Apply the variational unitary of a single layer to a state."""
        in_f = self.arch[layer - 1]
        ancilla = in_f
        state_vec = _qobj_to_np(state)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: Tensor) -> Tensor:
            # Prepare the input state on the first in_f qubits
            qml.StatePrep(state_vec, wires=range(in_f))
            # Parameterised single‑qubit rotations on all (in_f + 1) qubits
            for i in range(in_f + 1):
                qml.RY(params[3 * i], wires=i)
                qml.RZ(params[3 * i + 1], wires=i)
                qml.RX(params[3 * i + 2], wires=i)
            # Entangle ancilla with each input qubit
            for i in range(in_f):
                qml.CNOT(wires=[i, ancilla])
            # Return the full state vector
            return qml.state()

        new_state_vec = circuit(self.params[layer][output_idx])
        new_state = _np_to_qobj(new_state_vec, [2] * (in_f + 1))
        # Partial trace out the ancilla to keep only the input qubits
        return new_state.ptrace(range(in_f))

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, sample: qt.Qobj) -> List[qt.Qobj]:
        """Return the list of states for each layer."""
        states: List[qt.Qobj] = [sample]
        current = sample
        for layer in range(1, len(self.arch)):
            current = self._apply_layer_qml(current, layer, 0)
            states.append(current)
        return states

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    def train(
        self,
        training_data: Iterable[Tuple[qt.Qobj, qt.Qobj]],
        epochs: int = 100,
        lr: float = 0.01,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = torch.nn.functional.mse_loss,
        graph: nx.Graph | None = None,
        reg_weight: float = 0.0,
        verbose: bool = False,
    ) -> List[float]:
        """Perform a gradient‑based optimisation of the variational parameters."""
        opt = torch.optim.Adam(list(self.params.values()), lr=lr)
        loss_hist: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample, target in training_data:
                opt.zero_grad()
                states = self.forward(sample)
                pred_vec = _state_vector_tensor(states[-1])
                target_vec = _state_vector_tensor(target)
                loss = loss_fn(pred_vec, target_vec)
                if graph is not None:
                    state_vecs = [
                        _state_vector_tensor(s) for s in states[1:]
                    ]
                    loss += reg_weight * graph_regularizer(state_vecs, graph)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(training_data)
            loss_hist.append(epoch_loss)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
        return loss_hist

    # ---------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------
    def get_params(self) -> dict[int, Tensor]:
        """Return the current parameter tensors."""
        return self.params

    def set_params(self, params: dict[int, Tensor]) -> None:
        """Set the parameters of the network."""
        for layer, new_p in params.items():
            self.params[layer].copy_(new_p)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "HybridGraphQNNQML",
]
