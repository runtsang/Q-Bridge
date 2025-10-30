"""GraphQNN__gen189: Quantum graph neural network with variational training.

This module mirrors the classical version but uses PennyLane to define
a parameterized quantum circuit.  The network is trained by minimizing
the negative fidelity between the circuit output and a target unitary
applied to random input states.  The class provides a simple training
loop and a method to construct a fidelity‑based adjacency graph.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Generate a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    # Use QR decomposition of a random complex matrix
    random_matrix = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    q, r = pnp.linalg.qr(random_matrix)
    # Ensure determinant 1 (special unitary)
    d = pnp.diagonal(r)
    ph = d / pnp.abs(d)
    return q @ pnp.diag(1 / ph)


def _random_qubit_state(num_qubits: int) -> Tensor:
    """Sample a random pure state on num_qubits qubits."""
    dim = 2 ** num_qubits
    vec = pnp.random.randn(dim) + 1j * pnp.random.randn(dim)
    vec /= pnp.linalg.norm(vec)
    return vec


def random_training_data(
    unitary: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input_state, target_state) pairs."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        inp = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        tgt = unitary @ inp
        dataset.append((inp, tgt))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], np.ndarray, List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Build a parameterized quantum circuit.

    Parameters:
        qnn_arch: list where all entries are equal to the qubit count.
                  The length of the list minus one is the number of layers.
        samples: number of training samples.

    Returns:
        architecture,
        initial parameters (flattened array),
        training data,
        target unitary (ground truth)
    """
    # Sanity check: all layers must have the same qubit count
    if len(set(qnn_arch))!= 1:
        raise ValueError("All layers must have the same qubit count for this ansatz.")
    num_qubits = qnn_arch[0]
    num_layers = len(qnn_arch) - 1

    # Target unitary (fixed, non‑learnable)
    target = _random_qubit_unitary(num_qubits)

    # Training data
    training_data = random_training_data(target, samples)

    # Initial parameters: 3 rotations per qubit per layer
    params = pnp.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 3))
    params = params.reshape(-1)

    return list(qnn_arch), params, training_data, target


def _build_circuit(
    arch: List[int], params: np.ndarray
) -> qml.QNode:
    """Return a PennyLane QNode that implements the variational circuit."""
    num_qubits = arch[0]
    num_layers = len(arch) - 1
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(state_vec: Tensor, params: np.ndarray):
        # Prepare input state
        qml.QubitStateVector(state_vec, wires=range(num_qubits))
        # Unfold parameters
        idx = 0
        for layer in range(num_layers):
            for q in range(num_qubits):
                rz = params[idx]
                ry = params[idx + 1]
                rx = params[idx + 2]
                qml.RZ(rz, wires=q)
                qml.RY(ry, wires=q)
                qml.RX(rx, wires=q)
                idx += 3
            # Entangling layer: adjacent CNOTs
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return circuit


def feedforward(
    qnn_arch: Sequence[int],
    params: np.ndarray,
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the quantum circuit on each sample and collect the final state."""
    outputs: List[List[Tensor]] = []
    circuit = _build_circuit(list(qnn_arch), params)
    for inp, _ in samples:
        out = circuit(inp, params)
        outputs.append([out])  # only final state needed for fidelity
    return outputs


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return fidelity between two pure state vectors."""
    return float(pnp.abs(pnp.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state‑to‑state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen189:
    """Quantum graph neural network with variational training."""

    def __init__(self, qnn_arch: Sequence[int], samples: int = 100, lr: float = 0.01):
        self.arch, self.params, self.training_data, self.target = random_network(
            qnn_arch, samples
        )
        self.optimizer = qml.AdamOptimizer(stepsize=lr)

    def _loss(self, params: np.ndarray) -> float:
        """Negative average fidelity over the training set."""
        loss = 0.0
        for inp, tgt in self.training_data:
            out = _build_circuit(self.arch, params)(inp, params)
            loss += 1.0 - state_fidelity(out, tgt)
        return loss / len(self.training_data)

    def train(self, epochs: int = 200) -> None:
        """Train the variational circuit by minimizing negative fidelity."""
        for epoch in range(epochs):
            self.params = self.optimizer.step(self._loss, self.params)
            if epoch % 20 == 0 or epoch == epochs - 1:
                loss_val = self._loss(self.params)
                print(f"Epoch {epoch:3d}/{epochs:3d}  loss={loss_val:.4f}")

    def evaluate(
        self,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph of the circuit outputs."""
        outputs = [
            _build_circuit(self.arch, self.params)(inp, self.params)
            for inp, _ in self.training_data
        ]
        return fidelity_adjacency(outputs, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def get_arch(self) -> List[int]:
        return list(self.arch)

    def get_params(self) -> np.ndarray:
        return self.params

    def get_target(self) -> Tensor:
        return self.target


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen189",
]
