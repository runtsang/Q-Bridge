"""GraphQNN__gen208: quantum implementation using Pennylane.

This module mirrors the classical version but uses a variational quantum circuit.
The interface is identical so the two implementations can be swapped at runtime.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qnode
from pennylane import default_qubit

Tensor = np.ndarray


class GraphQNN__gen208:
    """Quantum graph neural network implemented with Pennylane.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes.  Each layer corresponds to a set of output qubits that
        receive rotations from the input qubits plus an ancilla.
    dev : pennylane.Device, optional
        Quantum device (default: default_qubit with enough wires).
    """

    def __init__(self, arch: Sequence[int], dev: qml.Device | None = None):
        self.arch = list(arch)
        self.num_qubits = max(arch) + 1  # +1 for ancilla per layer
        self.dev = dev if dev is not None else qml.device("default.qubit", wires=self.num_qubits)
        self.params = self._random_params()

    # ------------------------------------------------------------------ #
    # 1. Parameter handling
    # ------------------------------------------------------------------ #
    def _random_params(self) -> np.ndarray:
        """Return a flat array of rotation angles for all layers."""
        params = []
        for layer_idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            # For each output qubit we need a rotation for each input qubit plus one ancilla.
            num_params_per_gate = 3
            params.append(pnp.random.uniform(-np.pi, np.pi, size=(out_f, in_f + 1, num_params_per_gate)))
        return np.concatenate([p.flatten() for p in params])

    def _reshape_params(self, flat_params: np.ndarray) -> List[np.ndarray]:
        """Reshape the flat parameter vector into per‑layer tensors."""
        reshaped = []
        idx = 0
        for layer_idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            size = out_f * (in_f + 1) * 3
            layer_params = flat_params[idx : idx + size].reshape(out_f, in_f + 1, 3)
            reshaped.append(layer_params)
            idx += size
        return reshaped

    # ------------------------------------------------------------------ #
    # 2. Quantum circuit
    # ------------------------------------------------------------------ #
    def _circuit(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply the variational ansatz to the input state."""
        qml.StatePrep(state, wires=range(self.num_qubits))
        layer_params = self._reshape_params(params)
        for layer_idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            for out_q in range(out_f):
                for in_q in range(in_f):
                    # Apply rotations from input qubit to output qubit
                    angles = layer_params[out_q, in_q, :]
                    qml.Rot(*angles, wires=[in_q, out_q + in_f])
                # Entangle with ancilla
                qml.CNOT(wires=[out_q + in_f, out_q + in_f + 1])
        return qml.state()

    @qnode
    def quantum_forward(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Pennylane QNode that returns the state vector after the ansatz."""
        return self._circuit(state, params)

    # ------------------------------------------------------------------ #
    # 3. Data generation helpers
    # ------------------------------------------------------------------ #
    def random_training_data(self, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate random input–output pairs."""
        data: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            # Random pure state on input qubits
            dim = 2 ** self.arch[0]
            vec = pnp.random.randn(dim) + 1j * pnp.random.randn(dim)
            vec /= pnp.linalg.norm(vec)
            state = np.zeros(2 ** self.num_qubits, dtype=complex)
            state[: dim] = vec
            # Target state: apply a random unitary to the input part
            U = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
            U = pnp.linalg.qr(U)[0]
            target_vec = U @ vec
            target_state = np.zeros(2 ** self.num_qubits, dtype=complex)
            target_state[: dim] = target_vec
            data.append((state, target_state))
        return data

    def random_network(self, samples: int) -> Tuple[List[int], np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Return architecture, parameters, training data and target unitary."""
        params = self._random_params()
        training_data = self.random_training_data(samples)
        target_unitary = pnp.random.randn(2 ** self.arch[-1], 2 ** self.arch[-1]) + 1j * pnp.random.randn(2 ** self.arch[-1], 2 ** self.arch[-1])
        target_unitary = pnp.linalg.qr(target_unitary)[0]
        return self.arch, params, training_data, target_unitary

    # ------------------------------------------------------------------ #
    # 4. Forward pass
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Return the state after each layer for each sample."""
        outputs: List[List[np.ndarray]] = []
        for state, _ in samples:
            current = state
            layerwise = [current]
            flat_params = self.params
            for layer_idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
                # Apply the whole circuit; for simplicity we just run the full forward
                current = self.quantum_forward(current, flat_params)
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    # ------------------------------------------------------------------ #
    # 5. Fidelity helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared overlap of two pure states."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: Optional[float] = None,
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

    # ------------------------------------------------------------------ #
    # 6. Hybrid training step
    # ------------------------------------------------------------------ #
    def train_step(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
        opt: qml.GradientDescentOptimizer,
        loss_fn: callable = None,
        lr: float = 0.01,
    ) -> float:
        """One gradient step that optimizes the quantum parameters.

        The loss is a weighted sum of the mean‑square error between the
        output state and the target state.
        """
        if loss_fn is None:
            def loss_fn(params):
                loss = 0.0
                for state, target in samples:
                    out = self.quantum_forward(state, params)
                    loss += pnp.mean((out - target) ** 2)
                return loss / len(samples)
        # Use the provided optimizer to update parameters
        self.params = opt.step(self.params, loss_fn)
        return loss_fn(self.params)

    # ------------------------------------------------------------------ #
    # 7. Convenience wrappers
    # ------------------------------------------------------------------ #
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Return the output state for a single input."""
        return self.quantum_forward(state, self.params)

    __all__ = [
        "GraphQNN__gen208",
    ]
