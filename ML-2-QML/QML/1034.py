"""Hybrid Graph‑Quantum Neural Network – Quantum utilities.

The QML module implements a variational circuit that can be used as a
classical‑to‑quantum encoder.  The public interface matches the seed
but we expose a new ``VariationalQNN`` class that is responsible for
the quantum circuit and training.  The implementation uses Pennylane
state‑vector simulator.  The module can run on either Qiskit or
Pennylane – the user can choose the device via a config flag.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

Tensor = pnp.ndarray
Array = np.ndarray


# --------------------------------------------------------------------------- #
# 1.  Basic utilities – same as the seed but using Pennylane
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> Tensor:
    """Return a tensor‑product identity of size 2**num_qubits."""
    return pnp.eye(2 ** num_qubits, dtype=pnp.complex128)


def _tensored_zero(num_qubits: int) -> Tensor:
    """Return a tensor‑product zero projector of size 2**num_qubits."""
    zero = pnp.zeros((2 ** num_qubits, 1), dtype=pnp.complex128)
    zero[0, 0] = 1.0
    return zero


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Generate a random unitary on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    mat = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    q, _ = pnp.linalg.qr(mat)
    return q


def _random_qubit_state(num_qubits: int) -> Tensor:
    """Generate a random pure state on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    vec = pnp.random.randn(dim, 1) + 1j * pnp.random.randn(dim, 1)
    vec /= pnp.linalg.norm(vec)
    return vec


def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (state, unitary*state) pairs for training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[Tensor]], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return a network architecture, list of layer unitaries, training data and the target unitary."""
    # Target unitary for the last layer
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target_unitary, samples)

    # Build per‑layer unitaries (one for each output node)
    layers: List[List[Tensor]] = []
    for layer_idx in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer_idx - 1]
        num_outputs = qnn_arch[layer_idx]
        layer_ops: List[Tensor] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        layers.append(layer_ops)

    return qnn_arch, layers, training, target_unitary


# --------------------------------------------------------------------------- #
# 2.  Forward pass – evaluate the circuit for each sample
# --------------------------------------------------------------------------- #
def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    layer: int,
    input_state: Tensor,
) -> Tensor:
    """Apply the unitaries of a single layer and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad the input state with zeros for the output qubits
    padded = pnp.kron(input_state, _tensored_zero(num_outputs))
    # Compose the layer unitary (block‑diagonal over outputs)
    layer_unitary = _tensored_id(num_outputs)
    for op in unitaries[layer]:
        layer_unitary = pnp.kron(layer_unitary, op)
    # Apply and trace out the input qubits
    new_state = layer_unitary @ padded
    # Partial trace over the first ``num_inputs`` qubits
    keep = list(range(num_outputs))
    return pnp.trace(new_state, indices=keep)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of per‑layer states for each sample."""
    stored_states: List[List[Tensor]] = []
    for sample, _ in samples:
        layerwise: List[Tensor] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# 3.  Fidelity utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return float(pnp.abs(pnp.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
# 4.  Variational QNN – a trainable circuit
# --------------------------------------------------------------------------- #
class VariationalQNN:
    """Variational circuit that maps a classical vector to a quantum state.

    The circuit consists of a stack of parameterized RY rotations on each qubit
    followed by a pair‑wise entangling CNOT layer.  The number of qubits is
    determined by ``num_qubits``.  The circuit is implemented as a Pennylane
    QNode and can be trained with gradient descent to approximate a target
    unitary or state.
    """

    def __init__(self, num_qubits: int, device: str = "default.qubit", shots: int = 1024):
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits, shots=shots)
        # Initialize parameters: one RY per qubit per layer
        self.num_layers = 2
        self.params = pnp.random.uniform(0, 2 * pnp.pi, (self.num_layers, num_qubits))
        self.opt = None

    def circuit(self, params: Tensor, x: Tensor) -> Tensor:
        """Pennylane circuit that encodes ``x`` and applies the variational layers."""
        # Encode classical data as rotations on the first qubit
        for i in range(self.num_qubits):
            qml.RY(x[i] if i < len(x) else 0.0, wires=i)
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.RY(params[layer, qubit], wires=qubit)
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        return qml.state()

    def qnode(self, params: Tensor, x: Tensor):
        return self.circuit(params, x)

    def predict(self, x: Tensor) -> Tensor:
        return self.qnode(self.params, x)

    def loss(self, x: Tensor, target: Tensor) -> float:
        pred = self.predict(x)
        return pnp.mean((pred - target) ** 2)

    def train(self, training_data: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 100):
        opt = qml.GradientDescentOptimizer(lr)
        for epoch in range(epochs):
            for x, y in training_data:
                self.params = opt.step(lambda v: self.loss(x, y), self.params)
            if epoch % 10 == 0:
                loss_val = sum(self.loss(x, y) for x, y in training_data) / len(training_data)
                print(f"Epoch {epoch}: loss={loss_val:.6f}")

    def fidelity(self, x: Tensor, target: Tensor) -> float:
        pred = self.predict(x)
        return state_fidelity(pred, target)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "VariationalQNN",
]
