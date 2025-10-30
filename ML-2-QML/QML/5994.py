"""Utilities for building a variational quantum neural network with
parameter‑shift differentiable circuits and fidelity‑based adjacency graphs.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> np.ndarray:
    """Return a Haar‑random unitary matrix for `num_qubits`."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training pairs (|ψ⟩, U|ψ⟩)."""
    data = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        # random pure state
        vec = np.random.randn(unitary.shape[0]) + 1j * np.random.randn(unitary.shape[0])
        vec /= np.linalg.norm(vec)
        data.append((vec, unitary @ vec))
    return data


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generate a random variational network and training data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # For each layer create a random unitary that maps the previous layer size
    # to the next layer size.  We implement each as a full unitary on the
    # concatenated input+output registers and then trace out the inputs.
    unitaries: List[np.ndarray] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        dim = 2 ** (in_f + out_f)
        unitaries.append(_random_unitary(dim))
    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Return the reduced density matrix of `state` keeping qubits in `keep`."""
    dim = int(np.log2(state.shape[0]))
    # Build density matrix
    rho = np.outer(state, np.conj(state))
    # Trace out all qubits not in `keep`
    trace_out = [i for i in range(dim) if i not in keep]
    for qubit in sorted(trace_out, reverse=True):
        rho = np.trace(rho.reshape(2 ** (dim - 1), 2, 2 ** (dim - 1), 2), axis1=1, axis2=3)
    return rho


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Propagate each sample through the variational network and return
    the state vector at every layer.
    """
    stored_states: List[List[np.ndarray]] = []
    for state, _ in samples:
        layerwise: List[np.ndarray] = [state]
        current = state
        for layer, U in enumerate(unitaries, start=1):
            # Apply unitary on input + output registers
            dim = int(np.log2(U.shape[0]))
            # Pad current state with zeros on the output qubits
            padded = np.zeros(2 ** dim, dtype=complex)
            padded[: current.shape[0]] = current
            # Apply unitary
            new_state = U @ padded
            # Trace out the input qubits
            keep = list(range(dim - qnn_arch[layer - 1], dim))
            current = _partial_trace(new_state, keep).flatten()
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def parameter_shift_gradient(
    circuit: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    observable: np.ndarray = np.array([[1, 0], [0, -1]]),
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the gradient of ⟨O⟩ w.r.t. each parameter using the
    parameter‑shift rule.  `circuit` should return a state vector.
    """
    grads = np.zeros_like(params)
    for idx in range(len(params)):
        perturbed_plus = params.copy()
        perturbed_minus = params.copy()
        perturbed_plus[idx] += shift
        perturbed_minus[idx] -= shift
        psi_plus = circuit(perturbed_plus)
        psi_minus = circuit(perturbed_minus)
        exp_plus = np.vdot(psi_plus, observable @ psi_plus).real
        exp_minus = np.vdot(psi_minus, observable @ psi_minus).real
        grads[idx] = 0.5 * (exp_plus - exp_minus)
    return grads


# --------------------------------------------------------------------------- #
#  GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Quantum variational network that propagates quantum states through a
    sequence of random unitaries.  The API mirrors the classical counterpart
    and adds a parameter‑shift gradient estimator.
    """

    def __init__(self, arch: Sequence[int], samples: int) -> None:
        self.arch = list(arch)
        self.samples = samples
        self.arch, self.unitaries, self.training_data, self.target = self._build_network()

    def _build_network(self) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        arch, unitaries, training_data, target = random_network(self.arch, self.samples)
        return arch, unitaries, training_data, target

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Propagate raw samples through the variational circuit."""
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(self, states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def parameter_shift_gradient(self, circuit: Callable[[np.ndarray], np.ndarray], params: np.ndarray, observable: np.ndarray = np.array([[1, 0], [0, -1]])) -> np.ndarray:
        """Return the gradient of ⟨O⟩ w.r.t. `params` using the parameter‑shift rule."""
        return parameter_shift_gradient(circuit, params, observable=observable)

    def __repr__(self) -> str:
        return f"GraphQNN(arch={self.arch}, samples={self.samples})"

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "parameter_shift_gradient",
]
