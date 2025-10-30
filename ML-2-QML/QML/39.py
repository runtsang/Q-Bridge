"""
GraphQNN__gen040_QML.py

Quantum version that builds on PennyLane to provide a variational
graph‑neural‑network interface.  The module keeps the same public API
as the seed but now supports automatic differentiation, a hybrid
loss with classical outputs, and a simple training routine that
updates the circuit parameters using Adam.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List, Sequence, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

Tensor = np.ndarray

def _random_qubit_unitary(num_qubits: int):
    """Return a parameterised variational circuit acting on ``num_qubits`` qubits."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(state, *params):
        # Encode the input state
        for i in range(num_qubits):
            qml.QubitStateVector(state[:, i], wires=i)
        # Apply a layer of RY rotations
        for i, p in enumerate(params):
            qml.RY(p, wires=i % num_qubits)
        return qml.state()

    return circuit, num_qubits

def random_training_data(unitary, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training data from a target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = np.random.randn(unitary.num_wires, 1)
        state = state / np.linalg.norm(state)
        target = unitary(state, *np.random.randn(unitary.num_wires))[0]
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tuple[qml.QNode, int]], List[Tuple[Tensor, Tensor]], qml.QNode]:
    """Create a random variational network and training data for its last layer."""
    target_circuit, _ = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_circuit, samples)

    unitaries: List[Tuple[qml.QNode, int]] = []
    for layer in range(1, len(qnn_arch)):
        circuit, num_params = _random_qubit_unitary(qnn_arch[layer])
        unitaries.append((circuit, num_params))

    return list(qnn_arch), unitaries, training_data, target_circuit

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Tuple[qml.QNode, int]], layer: int, input_state: Tensor) -> Tensor:
    """Apply a variational layer to the input state and return the output state."""
    circuit, num_params = unitaries[layer]
    params = np.random.randn(num_params)
    return circuit(input_state, *params)[0]

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Tuple[qml.QNode, int]], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return a list of state vectors for each sample and layer."""
    stored_states: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise: List[Tensor] = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Hybrid training loop
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Variational graph‑neural‑network using PennyLane.
    Provides a ``train`` method that optimises the circuit parameters
    with a hybrid loss mixing classical MSE and quantum fidelity.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.circuits: List[qml.QNode] = []
        self.params: List[np.ndarray] = []
        for l in arch[1:]:
            circuit, num_params = _random_qubit_unitary(l)
            self.circuits.append(circuit)
            self.params.append(np.random.randn(num_params))

    def forward(self, state: Tensor) -> Tensor:
        """Return the output of the last variational layer."""
        current = state
        for circuit, p in zip(self.circuits, self.params):
            current = circuit(current, *p)[0]
        return current

    def hybrid_loss(self, outputs: Tensor, targets: Tensor, quantum_states: Iterable[Tensor] | None = None, alpha: float = 0.5) -> float:
        """Weighted sum of MSE and fidelity loss."""
        mse = np.mean((outputs - targets) ** 2)
        if quantum_states is None:
            return mse
        fid_losses = []
        for out, qstate in zip(outputs, quantum_states):
            out_norm = out / np.linalg.norm(out)
            fid = np.abs(np.vdot(out_norm, qstate)) ** 2
            fid_losses.append(1.0 - fid)
        fid_loss = np.mean(fid_losses)
        return alpha * mse + (1 - alpha) * fid_loss

    def train_model(self,
                    training_data: List[Tuple[Tensor, Tensor]],
                    quantum_states: List[Tensor] | None = None,
                    epochs: int = 100,
                    lr: float = 0.01,
                    alpha: float = 0.5) -> List[float]:
        """Simple training loop using Adam optimiser from PennyLane."""
        opt = qml.AdamOptimizer(stepsize=lr)
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for state, target in training_data:
                def loss_fn(*params):
                    # Flatten all parameters
                    idx = 0
                    out = state
                    for circuit, p_len in zip(self.circuits, [len(p) for p in self.params]):
                        out = circuit(out, *params[idx:idx+p_len])[0]
                        idx += p_len
                    return self.hybrid_loss(out, target, [qstate] if quantum_states is not None else None, alpha=alpha)

                loss = loss_fn(*self.params)
                self.params = opt.step(loss, self.params)
                epoch_loss += loss
            loss_history.append(epoch_loss / len(training_data))
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} loss {loss_history[-1]:.6f}")
        return loss_history

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
