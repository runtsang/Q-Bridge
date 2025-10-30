"""GraphQNN – Quantum‑Variational QNN with circuit construction and fidelity‑based sampling.

This module re‑implements the core utilities from the classical seed but
adapts them to a variational quantum circuit built with Qiskit.  The
implementation focuses on the model structure and data generation; training
pipelines are left as placeholders to keep the module concise.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, random_statevector, random_unitary

Tensor = Statevector

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]

# --------------------------------------------------------------------------- #
#  Random data generation
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> Statevector:
    """Return a random unitary as a Statevector object."""
    dim = 2 ** num_qubits
    unitary = random_unitary(dim)
    return Statevector(unitary)


def _random_qubit_state(num_qubits: int) -> Statevector:
    """Return a random pure state as a Statevector object."""
    return random_statevector(2 ** num_qubits)


def random_training_data(
    unitary: Statevector, samples: int
) -> List[Tuple[Statevector, Statevector]]:
    """Generate synthetic data ``(state, unitary * state)``."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.dim)
        target = unitary.evolve(state)
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int, *, seed: Optional[int] = None):
    """Generate a random variational network.

    Returns
    -------
    arch : List[int]
        Architecture list.
    circuits : List[List[QuantumCircuit]]
        List of parameterised circuits per layer.
    training_data : List[Tuple[Statevector, Statevector]]
        Synthetic training data generated from the target unitary.
    target_unitary : Statevector
        The unitary that the variational circuit is trained to approximate.
    """
    if seed is not None:
        np.random.seed(seed)

    # Target unitary – a random unitary on the last layer size
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_circuits: List[QuantumCircuit] = []
        for out_idx in range(num_outputs):
            qc = QuantumCircuit(num_inputs + 1)
            dim = 2 ** (num_inputs + 1)
            unitary_matrix = random_unitary(dim)
            qc.unitary(unitary_matrix, qc.qubits, label=f"U_{layer}_{out_idx}")
            layer_circuits.append(qc)
        circuits.append(layer_circuits)

    return list(qnn_arch), circuits, training_data, target_unitary


# --------------------------------------------------------------------------- #
#  Partial trace helpers (statevector version)
# --------------------------------------------------------------------------- #

def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the reduced statevector keeping the specified qubits."""
    return state.ptrace(keep)


def _partial_trace_remove(state: Statevector, remove: Sequence[int]) -> Statevector:
    """Return the reduced statevector after removing the specified qubits."""
    keep = [i for i in range(state.num_qubits) if i not in remove]
    return state.ptrace(keep)


# --------------------------------------------------------------------------- #
#  Layer channel – apply a single layer of the variational circuit
# --------------------------------------------------------------------------- #

def _layer_channel(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    """Apply one layer of the variational circuit to an input state."""
    num_inputs = qnn_arch[layer - 1]
    # Use the first circuit for the layer (single‑output simplification)
    gate = circuits[layer][0]
    zero_state = Statevector.from_label("0")
    combined = input_state.tensor(zero_state)
    layer_unitary = gate.evolve(combined)
    return _partial_trace_remove(layer_unitary, list(range(num_inputs)))


# --------------------------------------------------------------------------- #
#  Forward pass
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Run a forward pass through a variational QNN.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture list.
    circuits : Sequence[Sequence[QuantumCircuit]]
        List of circuits per layer.
    samples : Iterable[Tuple[Statevector, Statevector]]
        Iterable of input/target pairs. Only the input part is used.

    Returns
    -------
    List[List[Statevector]]
        Statevectors per sample, including the input as the first element.
    """
    activations: List[List[Statevector]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, circuits, layer, current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


# --------------------------------------------------------------------------- #
#  Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the squared magnitude of the inner product."""
    return float(abs(np.vdot(a.data, b.data)) ** 2)


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities.

    Parameters
    ----------
    states : Sequence[Statevector]
        List of statevectors.
    threshold : float
        Primary fidelity threshold.
    secondary : float, optional
        Secondary threshold for additional edges.
    secondary_weight : float, default 0.5
        Weight for secondary edges.

    Returns
    -------
    nx.Graph
        Weighted graph where nodes correspond to the indices of ``states``.
    """
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
#  GraphQNN class – quantum wrapper
# --------------------------------------------------------------------------- #

class GraphQNN:
    """Quantum variational graph neural network.

    The class mirrors the classical :class:`GraphQNN` but uses Qiskit to
    construct a parameterised circuit that maps an input state to an output
    state.  The training routine is intentionally lightweight – only a
    placeholder for an optimizer is provided.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture list.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, arch: Sequence[int], seed: Optional[int] = None):
        self.arch = list(arch)
        self.seed = seed
        self.circuits: List[List[QuantumCircuit]] = []
        self.build_circuits()

    # --------------------------------------------------------------------- #
    #  Circuit construction
    # --------------------------------------------------------------------- #

    def build_circuits(self) -> None:
        """Generate a fresh random circuit for each layer."""
        if self.seed is not None:
            np.random.seed(self.seed)
        self.circuits = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_circuits: List[QuantumCircuit] = []
            for out_idx in range(num_outputs):
                qc = QuantumCircuit(num_inputs + 1)
                dim = 2 ** (num_inputs + 1)
                unitary_matrix = random_unitary(dim)
                qc.unitary(unitary_matrix, qc.qubits, label=f"U_{layer}_{out_idx}")
                layer_circuits.append(qc)
            self.circuits.append(layer_circuits)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, input_state: Statevector) -> Statevector:
        """Apply the full variational circuit to an input state."""
        current = input_state
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.circuits, layer, current)
        return current

    # --------------------------------------------------------------------- #
    #  Training placeholder
    # --------------------------------------------------------------------- #

    def train(
        self,
        training_data: List[Tuple[Statevector, Statevector]],
        epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> None:
        """Very light‑weight training loop using finite‑difference gradients.

        The method is intentionally minimal and serves only as an example.
        """
        # Placeholder – user should replace with a proper optimizer
        for _ in range(epochs):
            for inp, tgt in training_data:
                # Forward
                out = self.forward(inp)
                # Compute fidelity loss
                loss = 1 - state_fidelity(out, tgt)
                # Back‑prop not implemented – this is a stub
                _ = loss  # silence unused variable warning
