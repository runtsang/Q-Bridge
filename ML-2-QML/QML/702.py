"""Utilities for building graph-based quantum neural networks.

This module extends the original QML seed by adding a variational circuit
implemented with Pennylane.  The ``train`` method performs a gradient‑based
optimization of the circuit parameters to maximise fidelity with a target
unitary.  The class can run on any Pennylane device (simulator or real backend).
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Any

import networkx as nx
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state /= np.linalg.norm(state)
    return state


def random_training_data(
    unitary: np.ndarray,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate dataset of (state, target_state) pairs."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target_state = unitary @ state
        dataset.append((state, target_state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random variational network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, target_unitary, training_data


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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


class GraphQNN__gen369:
    """Quantum graph neural network implemented with Pennylane.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer, e.g. ``[2, 3, 2]``.
    dev : str or qml.Device, optional
        Pennylane device.  Defaults to ``'default.qubit'`` with the
        number of qubits equal to the last element of ``arch``.
    """

    def __init__(self, arch: Sequence[int], dev: str | qml.Device | None = None):
        self.arch = list(arch)
        self.num_qubits = self.arch[-1]
        if dev is None:
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
        elif isinstance(dev, str):
            self.dev = qml.device(dev, wires=self.num_qubits)
        else:
            self.dev = dev
        self.params = nn.ParameterList()
        for layer in range(len(self.arch) - 1):
            out_f = self.arch[layer + 1]
            num_params = out_f * 3
            self.params.append(nn.Parameter(0.01 * torch.randn(num_params, dtype=torch.double)))

        def _circuit(params, input_state):
            qml.StatePrep(input_state, wires=range(self.num_qubits))
            offset = 0
            for layer_idx in range(len(self.arch) - 1):
                out_f = self.arch[layer_idx + 1]
                for q in range(out_f):
                    idx = offset + q * 3
                    qml.RX(params[layer_idx][idx], wires=q)
                    qml.RY(params[layer_idx][idx + 1], wires=q)
                    qml.RZ(params[layer_idx][idx + 2], wires=q)
                offset += out_f * 3
                for q in range(out_f - 1):
                    qml.CZ(wires=[q, q + 1])
            return qml.state()

        self.circuit = qml.qnode(self.dev, interface="torch")(_circuit)

    def forward(self, input_state: Tensor) -> Tensor:
        """Return the state vector after the variational circuit."""
        return self.circuit(self.params, input_state)

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[Tensor]:
        """Run the circuit on a batch of input states and return the outputs."""
        outputs: List[Tensor] = []
        for state, _ in samples:
            outputs.append(self.circuit(self.params, state))
        return outputs

    @staticmethod
    def train(
        model: "GraphQNN__gen369",
        training_data: List[Tuple[Tensor, Tensor]],
        optimizer: optim.Optimizer,
        epochs: int = 200,
        device: torch.device | None = None,
    ) -> Dict[str, List[float]]:
        """Train the variational circuit to match a target unitary.

        The loss is ``1 - fidelity`` between the circuit output and the target state.
        """
        if device is None:
            device = torch.device("cpu")
        history = {"loss": [], "fidelity": []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_fid = 0.0
            for state, target in training_data:
                state_t = torch.tensor(state, dtype=torch.double, device=device)
                target_t = torch.tensor(target, dtype=torch.double, device=device)
                optimizer.zero_grad()
                output = model.circuit(model.params, state_t)
                fidelity = torch.abs(torch.vdot(output, target_t)) ** 2
                loss = 1.0 - fidelity
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_fid += fidelity.item()
            epoch_loss /= len(training_data)
            epoch_fid /= len(training_data)
            history["loss"].append(epoch_loss)
            history["fidelity"].append(epoch_fid)
        return history

    @staticmethod
    def plot_fidelity(fidelity_history: List[float]) -> None:
        """Plot fidelity over training epochs."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(fidelity_history, label="Fidelity")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title("Quantum Training Fidelity")
        plt.legend()
        plt.tight_layout()
        plt.show()


__all__ = [
    "GraphQNN__gen369",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
