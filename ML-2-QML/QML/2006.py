from __future__ import annotations

import itertools
import math
import numpy as np
from typing import Iterable, Sequence, List, Tuple, Optional

import networkx as nx
import pennylane as qml
import torch
import torch.optim as optim

Tensor = torch.Tensor
State = np.ndarray


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def state_fidelity(a: State, b: State) -> float:
    """Absolute squared overlap between two pure state vectors."""
    return float(abs(np.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[State],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(target_unitary: np.ndarray, samples: int) -> List[Tuple[State, State]]:
    """Generate a set of input–output state pairs for a target unitary."""
    data: List[Tuple[State, State]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        # Random pure state
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        target_state = target_unitary @ state
        data.append((state, target_state))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Create a random variational circuit architecture and a training set.

    qnn_arch is interpreted as [num_qubits, num_layers].
    """
    num_qubits, num_layers = qnn_arch[0], qnn_arch[1]
    target_unitary = qml.utils.random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    # Random initial parameters for the ansatz
    params = np.random.randn(num_layers, num_qubits, 3)
    return list(qnn_arch), params, training_data, target_unitary


def feedforward(
    qnn_arch: Sequence[int],
    params: np.ndarray,
    samples: Iterable[Tuple[State, State]],
) -> List[State]:
    """Run the variational circuit on each input state and return the output state."""
    num_qubits, num_layers = qnn_arch[0], qnn_arch[1]
    dev = qml.device("default.qubit", wires=range(num_qubits))

    @qml.qnode(dev, interface="torch")
    def circuit(state, params):
        # Encode the input state
        qml.QubitStateVector(torch.tensor(state, dtype=torch.complex64), wires=range(num_qubits))
        # Ansatz
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)
            # Entangling layer
            for qubit in range(num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        return qml.state()

    outputs: List[State] = []
    for state, _ in samples:
        out = circuit(state, torch.tensor(params, dtype=torch.float32))
        outputs.append(out.detach().cpu().numpy())
    return outputs


class GraphQNN__gen206:
    """
    Hybrid quantum‑classical Graph‑based QNN that optimises a variational circuit
    to approximate a target unitary.  The circuit is parameterised by a
    3‑parameter rotation per qubit per layer and a simple CNOT entangling
    pattern.
    """

    def __init__(
        self,
        arch: Sequence[int],
        params: np.ndarray,
        target_state: State,
        training_data: List[Tuple[State, State]],
    ):
        self.arch = list(arch)
        self.num_qubits, self.num_layers = arch[0], arch[1]
        self.params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        self.target_state = torch.tensor(target_state, dtype=torch.complex64)
        self.training_data = training_data
        self.dev = qml.device("default.qubit", wires=range(self.num_qubits))

    # ------------------------------------------------------------------
    # Circuit definition
    # ------------------------------------------------------------------
    def _circuit(self, state: State):
        @qml.qnode(self.dev, interface="torch")
        def circuit_fn(state):
            qml.QubitStateVector(
                torch.tensor(state, dtype=torch.complex64), wires=range(self.num_qubits)
            )
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    qml.RX(self.params[layer, qubit, 0], wires=qubit)
                    qml.RY(self.params[layer, qubit, 1], wires=qubit)
                    qml.RZ(self.params[layer, qubit, 2], wires=qubit)
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.state()

        return circuit_fn(state)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def fidelity_loss(self, pred_state: torch.Tensor) -> torch.Tensor:
        pred_norm = pred_state / (torch.norm(pred_state) + 1e-12)
        target_norm = self.target_state / (torch.norm(self.target_state) + 1e-12)
        return 1.0 - torch.abs(torch.dot(pred_norm, target_norm)).pow(2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int = 200,
        lr: float = 0.01,
        early_stop_patience: int = 15,
    ) -> List[float]:
        """
        Optimise the circuit parameters to minimise the fidelity loss against the
        supplied target state.
        """
        optimizer = optim.Adam([self.params], lr=lr)
        best_loss = math.inf
        best_params = None
        patience_counter = 0
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for input_state, _ in self.training_data:
                optimizer.zero_grad()
                pred = self._circuit(input_state)
                loss = self.fidelity_loss(pred)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.training_data)
            history.append(epoch_loss)

            # Early‑stopping
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_params = self.params.clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

        if best_params is not None:
            self.params.data = best_params

        return history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen206",
]
