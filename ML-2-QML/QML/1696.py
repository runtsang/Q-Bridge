"""GraphQNN__gen275: Quantum graph neural network variant.

This module mirrors the classical interface but replaces the GCN backbone
with a variational quantum circuit implemented with Pennylane.
The quantum layer acts on a concatenated latent vector derived from a simple
fully‑connected encoder.  Fidelity based adjacency remains unchanged.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray

def _num_qubits_from_dim(dim: int) -> int:
    """Return the minimal number of qubits needed to encode a vector of length dim."""
    return int(np.ceil(np.log2(dim)))

def _encode_state(state: Tensor, num_qubits: int) -> Tensor:
    """Amplitude encode a vector into a quantum state of num_qubits."""
    vec = state.copy()
    target_len = 2 ** num_qubits
    if len(vec) < target_len:
        vec = np.concatenate([vec, np.zeros(target_len - len(vec))])
    return vec / np.linalg.norm(vec)

class GraphQNN__gen275:
    """Variational quantum graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture specifying the size of the latent vector at each
        layer.  The last element determines the number of qubits for the
        final unitary.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.num_qubits = _num_qubits_from_dim(arch[-1])
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        rng = np.random.default_rng()
        self.params: List[np.ndarray] = [
            rng.normal(size=(self.num_qubits, 3))
            for _ in range(len(arch) - 1)
        ]

    def _qnode(self, params_layer: np.ndarray):
        """Return a Pennylane QNode for the given parameters."""

        @qml.qnode(self.dev, interface="autograd")
        def circuit(state_vec: Tensor) -> Tensor:
            # State preparation
            qml.QubitStateVector(state_vec, wires=range(self.num_qubits))
            # Apply variational rotations
            for wire in range(self.num_qubits):
                qml.RX(params_layer[wire, 0], wires=wire)
                qml.RY(params_layer[wire, 1], wires=wire)
                qml.RZ(params_layer[wire, 2], wires=wire)
            # Entanglement
            for wire in range(self.num_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            return qml.state()
        return circuit

    def forward(self, state: Tensor) -> List[Tensor]:
        """Apply the variational circuit to a single state vector.

        Returns a list containing the state after each layer and the final
        output state.
        """
        state_vec = _encode_state(state, self.num_qubits)
        outputs: List[Tensor] = [state_vec]
        for layer_params in self.params:
            circuit = self._qnode(layer_params)
            state_vec = circuit(state_vec)
            outputs.append(state_vec)
        return outputs

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run the quantum circuit on a batch of samples.

        Each sample is a pair (input_state, target_state).  The target is
        unused in this forward pass.
        """
        results: List[List[Tensor]] = []
        for inp, _ in samples:
            results.append(self.forward(inp))
        return results

    def random_network(
        self,
        samples: int,
        seed: int | None = None,
    ) -> Tuple[List[int], List[np.ndarray], List[Tuple[Tensor, Tensor]], List[np.ndarray]]:
        """Generate a random set of variational parameters and training data."""
        rng = np.random.default_rng(seed)
        target_params = [
            rng.normal(size=(self.num_qubits, 3))
            for _ in range(len(self.arch) - 1)
        ]
        qnn = GraphQNN__gen275(self.arch)
        qnn.params = target_params
        training_data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            inp = rng.normal(size=self.arch[0])
            inp = inp / np.linalg.norm(inp)
            out_vec = qnn.forward(inp)[-1]
            training_data.append((inp, out_vec))
        return self.arch, target_params, training_data, target_params

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Return absolute squared overlap of two pure states."""
        return np.abs(np.vdot(a, b)) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# --------------------------------------------------------------------------- #
#  Backwards compatible functions
# --------------------------------------------------------------------------- #
def random_training_data(
    unitary_params: List[np.ndarray],
    samples: int,
    seed: int | None = None,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs for the quantum model."""
    rng = np.random.default_rng(seed)
    training_data: List[Tuple[Tensor, Tensor]] = []
    dim = 2 ** len(unitary_params[0])
    for _ in range(samples):
        inp = rng.normal(size=dim)
        inp = inp / np.linalg.norm(inp)
        qnn = GraphQNN__gen275([dim])
        qnn.params = unitary_params
        out = qnn.forward(inp)[-1]
        training_data.append((inp, out))
    return training_data

def random_network(
    arch: Sequence[int],
    samples: int,
    seed: int | None = None,
) -> Tuple[List[int], List[np.ndarray], List[Tuple[Tensor, Tensor]], List[np.ndarray]]:
    """Generate a random QNN architecture, training data and target unitary."""
    qnn = GraphQNN__gen275(arch)
    return qnn.random_network(samples, seed)

def feedforward(
    arch: Sequence[int],
    params: List[np.ndarray],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward pass using supplied parameters."""
    qnn = GraphQNN__gen275(arch)
    qnn.params = params
    return qnn.feedforward(samples)

def state_fidelity(a: Tensor, b: Tensor) -> float:
    return np.abs(np.vdot(a, b)) ** 2

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

__all__ = [
    "GraphQNN__gen275",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
