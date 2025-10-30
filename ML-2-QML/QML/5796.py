from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import numpy as npnp

Tensor = npnp.ndarray

# --------------------------------------------------------------------------- #
# 1. Helper functions – random states and unitaries
# --------------------------------------------------------------------------- #
def _random_qubit_state(num_qubits: int) -> Tensor:
    """Generate a random pure state vector of dimension 2**num_qubits."""
    dim = 2 ** num_qubits
    vec = npnp.random.randn(dim) + 1j * npnp.random.randn(dim)
    vec /= npnp.linalg.norm(vec)
    return vec

def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Generate a random unitary matrix via QR decomposition."""
    dim = 2 ** num_qubits
    mat = npnp.random.randn(dim, dim) + 1j * npnp.random.randn(dim, dim)
    q, r = np.linalg.qr(mat)
    d = np.diag(r)
    q = q * d / np.abs(d)
    return q

def random_training_data(
    unitary: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Create a dataset of input states and their target states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

# --------------------------------------------------------------------------- #
# 2. Variational ansatz – parameterised quantum circuit
# --------------------------------------------------------------------------- #
class GraphQNNAnsatz:
    """
    A shallow parameter‑efficient ansatz mirroring a classical feed‑forward
    network of depth ``len(arch)-1``.  Each layer applies a rotation RY on
    every qubit followed by an entangling CNOT chain.  The number of qubits
    equals the size of the last layer.
    """
    def __init__(self, arch: Sequence[int], seed: int | None = None):
        self.arch = list(arch)
        self.num_qubits = self.arch[-1]
        self.num_layers = len(arch) - 1
        if seed is not None:
            np.random.seed(seed)
        self.params = npnp.random.randn(self.num_layers, self.num_qubits)

        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev)
        def circuit(params: Tensor, state: Tensor) -> Tensor:
            qml.StatePrep(state, wires=range(self.num_qubits))
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RY(params[l, q], wires=q)
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()

        self._circuit = circuit

    def __call__(self, state: Tensor) -> Tensor:
        """Propagate a state through the ansatz."""
        return self._circuit(self.params, state)

# --------------------------------------------------------------------------- #
# 3. Random network construction
# --------------------------------------------------------------------------- #
def random_network(
    arch: List[int],
    samples: int,
) -> Tuple[List[int], GraphQNNAnsatz, List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Build a random ansatz and a corresponding training dataset.
    The target state for training is the output of the ansatz itself,
    i.e. we learn to reproduce the ansatz behaviour.
    """
    ansatz = GraphQNNAnsatz(arch)
    training_data = []
    for _ in range(samples):
        inp_state = _random_qubit_state(arch[-1])
        tgt_state = ansatz(inp_state)
        training_data.append((inp_state, tgt_state))
    target_state = ansatz(_random_qubit_state(arch[-1]))
    return arch, ansatz, training_data, target_state

# --------------------------------------------------------------------------- #
# 4. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Absolute squared overlap of two pure states."""
    return abs(npnp.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def compute_fidelity_graph(
    ansatz: GraphQNNAnsatz,
    dataset: Iterable[Tuple[Tensor, Tensor]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Run the ansatz on a dataset and build a fidelity graph."""
    states = [ansatz(state) for state, _ in dataset]
    return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
# 5. Feed‑forward helper
# --------------------------------------------------------------------------- #
def feedforward(
    arch: Sequence[int],
    ansatz: GraphQNNAnsatz,
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return input and output states for each sample."""
    return [[state, ansatz(state)] for state, _ in samples]

# --------------------------------------------------------------------------- #
# 6. Training loop – gradient descent on the ansatz
# --------------------------------------------------------------------------- #
def train_qnn(
    arch: Sequence[int],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
) -> GraphQNNAnsatz:
    """
    Train the parameterised ansatz to reproduce the target states.
    Loss is 1 – fidelity (equivalent to negative log‑likelihood for pure states).
    """
    ansatz = GraphQNNAnsatz(arch)
    opt = qml.AdamOptimizer(stepsize=lr)

    def loss_fn(params: Tensor, inp: Tensor, tgt: Tensor) -> float:
        pred = ansatz._circuit(params, inp)
        return 1.0 - state_fidelity(pred, tgt)

    for epoch in range(epochs):
        for inp, tgt in training_data:
            ansatz.params = opt.step(lambda p: loss_fn(p, inp, tgt), ansatz.params)
        if verbose and (epoch + 1) % 20 == 0:
            loss = np.mean([loss_fn(ansatz.params, inp, tgt) for inp, tgt in training_data])
            print(f"Epoch {epoch+1}/{epochs} loss {loss:.4f}")
    return ansatz

__all__ = [
    "GraphQNNAnsatz",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "compute_fidelity_graph",
    "train_qnn",
]
