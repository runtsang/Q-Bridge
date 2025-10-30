"""
GraphQNN__gen348: Quantum utilities for graph‑based variational circuits.

Key extensions
--------------
* Variational ansatz built from single‑qubit rotations and CNOT entanglers.
* PennyLane QNode that accepts a classical feature vector and outputs a state vector.
* Fidelity‑based loss that compares the circuit output to a target unitary.
* Training routine that optimizes the circuit parameters using autograd.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import scipy as sc

Tensor = pnp.ndarray


# --------------------------------------------------------------------------- #
# 1.  Variational ansatz construction
# --------------------------------------------------------------------------- #
def _cnot_entangler(num_qubits: int) -> List[Tuple[int, int]]:
    """Create a ring of CNOTs for nearest‑neighbour entanglement."""
    return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]


def build_ansatz(num_qubits: int, depth: int, device: str = "default.qubit") -> qml.QNode:
    """
    Return a PennyLane QNode that applies a depth‑thick variational circuit.
    Each layer consists of:
        * single‑qubit RY rotations with trainable angles.
        * a ring of CNOTs for entanglement.
    The input is a classical feature vector of length `num_qubits` and is encoded
    using a simple angle‑encoding scheme.
    """

    def circuit(x: Sequence[float], params: Sequence[float]) -> pnp.ndarray:
        # Angle‑encoding of the classical input
        for i, val in enumerate(x):
            qml.RY(val, wires=i)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i, j in _cnot_entangler(num_qubits):
                qml.CNOT(wires=[i, j])

        # Return the full state vector
        return qml.state()

    return qml.QNode(circuit, qml.Device(device, wires=num_qubits))


# --------------------------------------------------------------------------- #
# 2.  Quantum utilities mirroring the seed
# --------------------------------------------------------------------------- #
def random_training_data(target_unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """
    Generate synthetic data where the input is a random state and the target
    is the state after applying the target unitary.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amp /= sc.linalg.norm(amp)
        state = pnp.array(amp).flatten()
        target = target_unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """
    Generate a random target unitary and training data for the variational circuit.
    """
    num_qubits = qnn_arch[-1]
    target_unitary = pnp.random.rand(2 ** num_qubits, 2 ** num_qubits) + 1j * pnp.random.rand(
        2 ** num_qubits, 2 ** num_qubits
    )
    target_unitary = pnp.linalg.qr(target_unitary)[0]  # orthonormalize
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, target_unitary, training_data, target_unitary


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Overlap squared between two pure state vectors."""
    return float(abs(pnp.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
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
# 3.  Training routine
# --------------------------------------------------------------------------- #
def train_variational(
    qnode: qml.QNode,
    target_unitary: Tensor,
    training_data: List[Tuple[Tensor, Tensor]],
    epochs: int = 100,
    lr: float = 0.01,
    lambda_fid: float = 0.5,
) -> List[float]:
    """
    Optimise the parameters of `qnode` to minimise a hybrid loss:
        L = MSE(output, target) + λ * (1 - fidelity(output, target_unitary * input))
    The state returned by the QNode is compared to the target state.
    """
    params = qnode.trainable_params[0]
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    losses: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for state_in, state_target in training_data:
            def loss_fn(p):
                out = qnode(state_in, p)
                mse = pnp.mean((out - state_target) ** 2)
                fid = state_fidelity(out, target_unitary @ state_in)
                return mse + lambda_fid * (1.0 - fid)

            loss, grads = opt.step_and_cost(params, loss_fn)
            params = opt.apply_gradients(params, grads)
            epoch_loss += loss
        epoch_loss /= len(training_data)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.6f}")

    return losses


__all__ = [
    "build_ansatz",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
    "train_variational",
]
