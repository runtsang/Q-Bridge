import itertools
from typing import Iterable, Sequence, Tuple, List, Optional

import numpy as np
import pennylane as qml
import networkx as nx

Tensor = np.ndarray

def _random_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

def random_training_data(target_unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset of input and target states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim_in = target_unitary.shape[0]
    for _ in range(samples):
        state_in = np.random.randn(dim_in) + 1j * np.random.randn(dim_in)
        state_in /= np.linalg.norm(state_in)
        target = target_unitary @ state_in
        dataset.append((state_in, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network and corresponding training data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        dim = 2 ** qnn_arch[layer]
        unitary = _random_unitary(qnn_arch[layer])
        unitaries.append([unitary])
    return list(qnn_arch), unitaries, training_data, target_unitary

def _apply_layer(state: Tensor, unitary: Tensor, noise_rate: float = 0.0) -> Tensor:
    """Apply a unitary to a state vector with optional depolarising noise."""
    new_state = unitary @ state
    if noise_rate > 0.0 and np.random.rand() < noise_rate:
        noisy_unitary = _random_unitary(int(np.log2(new_state.shape[0])))
        new_state = noisy_unitary @ new_state
    return new_state

def _apply_network(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    input_state: Tensor,
    noise_rate: float = 0.0,
) -> Tensor:
    """Propagate a state through the full network."""
    current_state = input_state
    for layer in range(1, len(qnn_arch)):
        added = qnn_arch[layer] - qnn_arch[layer - 1]
        ancilla = np.zeros(2 ** added, dtype=complex)
        ancilla[0] = 1.0
        state_full = np.kron(current_state, ancilla)
        unitary = unitaries[layer][0]
        current_state = _apply_layer(state_full, unitary, noise_rate)
    return current_state

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    samples: Iterable[Tuple[Tensor, Tensor]],
    noise_rate: float = 0.0,
) -> List[List[Tensor]]:
    """Perform a forward pass through the network, storing intermediate states."""
    stored: List[List[Tensor]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            added = qnn_arch[layer] - qnn_arch[layer - 1]
            ancilla = np.zeros(2 ** added, dtype=complex)
            ancilla[0] = 1.0
            state_full = np.kron(current_state, ancilla)
            unitary = unitaries[layer][0]
            current_state = _apply_layer(state_full, unitary, noise_rate)
            layerwise.append(current_state)
        stored.append(layerwise)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared modulus of the inner product of two pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def train_qnn(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    target_unitary: Tensor,
    epochs: int = 10,
    lr: float = 0.01,
    noise_rate: float = 0.0,
) -> Sequence[Sequence[Tensor]]:
    """A simple finite‑difference gradient descent trainer with noise awareness."""
    for epoch in range(epochs):
        total_loss = 0.0
        for input_state, target_state in training_data:
            output_state = _apply_network(qnn_arch, unitaries, input_state, noise_rate)
            loss = np.linalg.norm(output_state - target_state) ** 2
            total_loss += loss
        total_loss /= len(training_data)
        print(f"Epoch {epoch+1}/{epochs} – loss: {total_loss:.6f}")

        # Finite‑difference gradient estimation
        for layer_idx in range(1, len(qnn_arch)):
            U = unitaries[layer_idx][0]
            grad = np.zeros_like(U)
            eps = 1e-5
            for r in range(U.shape[0]):
                for c in range(U.shape[1]):
                    U_plus = U.copy()
                    U_minus = U.copy()
                    U_plus[r, c] += eps
                    U_minus[r, c] -= eps
                    loss_plus = 0.0
                    loss_minus = 0.0
                    for inp, tgt in training_data:
                        out_plus = _apply_network(
                            qnn_arch,
                            unitaries[:layer_idx] + [[U_plus]] + unitaries[layer_idx+1:],
                            inp,
                            noise_rate,
                        )
                        loss_plus += np.linalg.norm(out_plus - tgt) ** 2
                        out_minus = _apply_network(
                            qnn_arch,
                            unitaries[:layer_idx] + [[U_minus]] + unitaries[layer_idx+1:],
                            inp,
                            noise_rate,
                        )
                        loss_minus += np.linalg.norm(out_minus - tgt) ** 2
                    loss_plus /= len(training_data)
                    loss_minus /= len(training_data)
                    grad[r, c] = (loss_plus - loss_minus) / (2 * eps)
            unitaries[layer_idx][0] = U - lr * grad
    return unitaries

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_qnn",
]
