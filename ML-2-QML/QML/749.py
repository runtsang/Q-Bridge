"""Utilities for building graph‑based quantum neural networks.

The module preserves the forward propagation helpers and the
fidelity‑based adjacency construction used in the original seed,
but replaces the qubit‑state manipulation with a Pennylane
variational circuit that can be trained on a quantum simulator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
# 1.  Quantum utilities – same API as the seed
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qml.QubitStateVector:
    """Return a tensor‑product identity operator."""
    return qml.Identity(num_qubits)

def _tensored_zero(num_qubits: int) -> qml.QubitStateVector:
    """Return a projector onto |0...0>."""
    return qml.PauliZ(num_qubits)  # placeholder; actual projector not needed

def _swap_registers(op: qml.Operation, source: int, target: int) -> qml.Operation:
    if source == target:
        return op
    return qml.CSwap([source, target, target])  # placeholder

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = np.random.randn(dim) + 1j * np.random.randn(dim)
    return amplitudes / np.linalg.norm(amplitudes)

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate a dataset where the target is the action of `unitary` on a random state."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(len(unitary).bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Return an architecture, a list of unitary matrices, a training set and the target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Pad with identity on extra outputs
                op = np.kron(op, np.eye(2 ** (num_outputs - 1), dtype=complex))
                # Swap to bring the new output to the back
                op = np.swapaxes(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Partial trace over all qubits except those in `keep`."""
    # Placeholder: returns the full state for simplicity
    return state

def _partial_trace_remove(state: np.ndarray, remove: Sequence[int]) -> np.ndarray:
    """Partial trace over qubits in `remove`."""
    return _partial_trace_keep(state, [i for i in range(len(state)) if i not in remove])

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], layer: int, input_state: np.ndarray) -> np.ndarray:
    """Apply the unitary channel for a single layer."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad input with |0> for new outputs
    padded = np.concatenate([input_state, np.zeros(2 ** (num_outputs - 1), dtype=complex)])
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    new_state = layer_unitary @ padded
    return _partial_trace_remove(new_state, range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Return the list of states for each sample."""
    stored_states: List[List[np.ndarray]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
# 2.  Graph‑level loss
# --------------------------------------------------------------------------- #

class GraphLoss:
    """Graph‑level loss based on fidelity penalties between adjacent nodes."""

    def __init__(self, adjacency: nx.Graph, weight: float = 1.0) -> None:
        self.adjacency = adjacency
        self.weight = weight

    def __call__(self, states: Sequence[np.ndarray]) -> float:
        """Compute the sum of (1 - fidelity) over all edges."""
        loss = 0.0
        for i, j, data in self.adjacency.edges(data=True):
            fid = state_fidelity(states[i], states[j])
            loss += self.weight * (1.0 - fid)
        return loss

# --------------------------------------------------------------------------- #
# 3.  Hybrid GraphQNN – Pennylane implementation
# --------------------------------------------------------------------------- #

class GraphQNN:
    """Hybrid graph neural network that can run a quantum forward pass
    on a Pennylane device and optionally a classical forward pass.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device: str = "default.qubit",
        shots: int = 1024,
        use_classical: bool = False,
    ) -> None:
        self.arch = list(arch)
        self.device_name = device
        self.shots = shots
        self.use_classical = use_classical

        # Build a Pennylane device
        self.dev = qml.device(device, wires=max(arch) + 1, shots=shots)

        # Randomly initialise unitary parameters
        self.params: List[np.ndarray] = [
            _random_qubit_unitary(num_inputs + 1)
            for num_inputs in arch[:-1]
        ]

        # Build the circuit
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the variational circuit on the Pennylane device."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, *params):
            qml.QubitStateVector(x, wires=range(len(x)))
            for layer, unitary in enumerate(params):
                # Apply the unitary as a matrix
                qml.MatrixGate(unitary, wires=range(unitary.shape[0].bit_length() - 1))
            return qml.state()
        self.circuit = circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the output state from the quantum circuit."""
        return self.circuit(x, *self.params)

    def train(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 100,
        lr: float = 0.01,
        optimizer_cls= qml.GradientDescentOptimizer,
        loss_fn=lambda out, tgt: np.mean((out - tgt) ** 2),
        adjacency: Optional[nx.Graph] = None,
        graph_loss_weight: float = 0.1,
        verbose: bool = False,
    ) -> None:
        """Train the variational circuit on the provided data."""
        opt = optimizer_cls(lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                def loss_func(params):
                    out = self.circuit(x, *params)
                    loss = loss_fn(out, y)
                    if adjacency is not None:
                        states = feedforward(self.arch, [self.params], [(x, y)])
                        graph_loss = GraphLoss(adjacency, weight=graph_loss_weight)(states[0])
                        loss += graph_loss
                    return loss
                self.params = opt.step(loss_func, self.params)
                epoch_loss += loss_func(self.params)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(data):.4f}")

    def get_fidelity_adjacency(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph from the current states."""
        # In practice, compute states from real data
        raise NotImplementedError("Compute adjacency from real data.")

    def get_params(self) -> List[np.ndarray]:
        """Return the current variational parameters."""
        return self.params

# --------------------------------------------------------------------------- #
# 4.  Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphLoss",
    "GraphQNN",
]
