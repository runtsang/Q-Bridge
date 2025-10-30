"""Quantum Graph Neural Network with depth‑aware variational ansatz.

The class mirrors the classical GraphQNN interface but operates on pure quantum states.
It uses Pennylane to build a parameterised circuit that applies a sequence of
single‑qubit rotations followed by a full‑SWAP entanglement layer.  The depth
is controlled by the ``architecture`` argument.  Training data consists of pairs
(state, target_state) where the target is a randomly generated unitary applied
to the input state.

The public API is compatible with the classical counterpart, making it
straight‑forward to compare fidelities across depth.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import pennylane as qml
import numpy as np

Tensor = np.ndarray


class GraphQNN:
    """Depth‑aware variational quantum circuit used as a QNN baseline."""

    def __init__(self, architecture: Sequence[int], layer_mask: Optional[List[int]] = None):
        """
        Parameters
        ----------
        architecture : Sequence[int]
            First element is the number of qubits. Subsequent elements
            are ignored but determine the depth via ``len(architecture)-1``.
        layer_mask : List[int], optional
            Indices of layers to keep active.  If ``None`` all layers are used.
        """
        self.num_qubits = architecture[0]
        self.layer_mask = list(range(1, len(architecture))) if layer_mask is None else layer_mask
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.params = self._init_params()
        self.target_unitary = self._random_unitary(self.num_qubits)

    def _init_params(self) -> List[np.ndarray]:
        """Randomly initialise rotation parameters for every layer."""
        params = []
        for _ in self.layer_mask:
            # Each qubit gets an (a,b,c) rotation
            params.append(np.random.randn(self.num_qubits, 3))
        return params

    @staticmethod
    def _random_unitary(num_qubits: int) -> np.ndarray:
        """Generate a random unitary on ``num_qubits`` qubits."""
        dim = 2**num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return q

    @staticmethod
    def random_training_data(
        target_unitary: np.ndarray,
        samples: int,
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate pairs (|ψ⟩, U_target|ψ⟩)."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        dim = target_unitary.shape[0]
        for _ in range(samples):
            vec = np.random.randn(dim) + 1j * np.random.randn(dim)
            vec /= np.linalg.norm(vec)
            target = target_unitary @ vec
            dataset.append((vec, target))
        return dataset

    @staticmethod
    def random_network(
        architecture: Sequence[int],
        samples: int,
        *,
        layer_mask: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[np.ndarray], List[Tuple[Tensor, Tensor]], np.ndarray]:
        """Convenient constructor that returns an architecture, parameters, data and target unitary."""
        qnn = GraphQNN(architecture, layer_mask)
        return (
            list(architecture),
            qnn.params,
            GraphQNN.random_training_data(qnn.target_unitary, samples),
            qnn.target_unitary,
        )

    def _variational_layer(self, params_arr: np.ndarray):
        """Single variational layer with rotations and full‑SWAP entanglement."""
        for i in range(self.num_qubits):
            qml.Rot(params_arr[i, 0], params_arr[i, 1], params_arr[i, 2], wires=i)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                qml.CNOT(wires=[i, j])
                qml.CNOT(wires=[j, i])
                qml.CNOT(wires=[i, j])

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        *,
        return_activations: bool = True,
    ) -> List[List[Tensor]]:
        """Return the state after each active layer for every sample."""
        stored: List[List[Tensor]] = []
        for inp, _ in samples:
            input_state = inp
            params_flat = np.concatenate([p.ravel() for p in self.params])

            def _circuit(params_flat):
                qml.QubitStateVector(input_state, wires=range(self.num_qubits))
                states = [input_state]
                idx = 0
                for p in self.params:
                    num_params = p.size
                    p_arr = params_flat[idx : idx + num_params].reshape(p.shape)
                    idx += num_params
                    self._variational_layer(p_arr)
                    states.append(qml.state())
                return states

            qnode = qml.QNode(_circuit, self.dev)
            states = qnode(params_flat)
            stored.append(states if return_activations else [states[-1]])
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared magnitude of inner product between two state vectors."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Weighted graph from fidelity matrix."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["GraphQNN"]
