"""GraphQNN__gen265 – Quantum graph neural network implemented with Qiskit.

The module mirrors the classical interface but operates on pure quantum
states.  It provides:
  * Variational circuit per layer with trainable RY parameters.
  * Fidelity‑based loss for learning a target unitary.
  * Utilities for generating random target unitaries and training data.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector, random_unitary
from qiskit.circuit.library import RY, CX

Tensor = np.ndarray


class GraphQNN__gen265:
    """Quantum graph neural network implemented with Qiskit.

    Architecture: list of layer widths.  Each layer is a variational
    unitary that acts on the first ``out`` qubits of the current state.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.n_qubits = self.arch[-1]
        # Parameters: one real angle per qubit per layer
        self.params: List[np.ndarray] = [
            np.random.uniform(0, 2 * np.pi, size=(self.arch[layer + 1],))
            for layer in range(len(self.arch) - 1)
        ]

    # ------------------------------------------------------------------
    #  Variational layer
    # ------------------------------------------------------------------
    def _apply_layer(self, state_vec: Tensor, params: Tensor, out_q: int) -> Tensor:
        """Apply a single variational layer to ``state_vec``."""
        sv = Statevector(state_vec)
        for i, theta in enumerate(params):
            sv = sv.apply(RY(theta, i))
        for i in range(out_q - 1):
            sv = sv.apply(CX(i, i + 1))
        return sv.data

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def feedforward(self, input_state: Tensor) -> List[Tensor]:
        """Return the state after each layer."""
        states: List[Tensor] = [input_state]
        current = input_state
        for layer_idx, layer_params in enumerate(self.params):
            out_q = self.arch[layer_idx + 1]
            current = self._apply_layer(current, layer_params, out_q)
            states.append(current)
        return states

    # ------------------------------------------------------------------
    #  Random data generation utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(
        unitary: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic (input, target) pairs."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            vec = np.random.randn(dim) + 1j * np.random.randn(dim)
            vec /= np.linalg.norm(vec)
            target = unitary @ vec
            dataset.append((vec, target))
        return dataset

    @staticmethod
    def random_network(
        arch: List[int], samples: int
    ) -> Tuple[List[int], Tensor, List[Tuple[Tensor, Tensor]], Tensor]:
        """Create a random target unitary and training data."""
        target_unitary = random_unitary(arch[-1])
        training_data = GraphQNN__gen265.random_training_data(target_unitary, samples)
        return arch, target_unitary, training_data, target_unitary

    # ------------------------------------------------------------------
    #  Fidelity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two pure states."""
        return abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen265.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Hybrid loss
    # ------------------------------------------------------------------
    @staticmethod
    def hybrid_loss(pred: Tensor, target: Tensor, fid_weight: float = 0.1) -> float:
        """Fidelity‑based loss for variational training."""
        fid = GraphQNN__gen265.state_fidelity(pred, target)
        return 1 - fid + fid_weight * (1 - fid)

    # ------------------------------------------------------------------
    #  Training routine (quantum only)
    # ------------------------------------------------------------------
    def train_quantum(
        self, data: Iterable[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 200
    ) -> None:
        """Optimize parameters to match the target unitary."""
        data = list(data)  # allow repeated passes
        for _ in range(epochs):
            for input_state, target_state in data:
                # Compute current prediction and loss
                pred = self.feedforward(input_state)[-1]
                loss = 1 - self.state_fidelity(pred, target_state)

                # Finite‑difference gradients
                grads: List[np.ndarray] = []
                for layer_idx, params in enumerate(self.params):
                    layer_grads = []
                    for i, theta in enumerate(params):
                        eps = 1e-5
                        params[i] += eps
                        new_pred = self.feedforward(input_state)[-1]
                        new_loss = 1 - self.state_fidelity(new_pred, target_state)
                        grad = (new_loss - loss) / eps
                        layer_grads.append(grad)
                        params[i] = theta
                    grads.append(np.array(layer_grads))
                # Parameter update
                for layer_idx, layer_grads in enumerate(grads):
                    self.params[layer_idx] -= lr * layer_grads


__all__ = [
    "GraphQNN__gen265",
]
