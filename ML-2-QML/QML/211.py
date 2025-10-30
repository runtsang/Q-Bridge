"""GraphQNN: quantum version with a variational circuit.

The module mirrors the classical API but replaces the linear layers
with a parameterised quantum circuit.  A simple training loop based
on the parameter‑shift rule is provided, and a fidelity‑based
adjacency graph is available for experimental analysis.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector

Tensor = np.ndarray

# --------------------------------------------------------------------------- #
# 1.  Utility functions
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # QR decomposition gives a unitary
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(
    target_unitary: np.ndarray,
    samples: int
) -> List[Tuple[Statevector, Statevector]]:
    """Create a dataset of input / target state pairs."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    n_qubits = int(np.log2(target_unitary.shape[0]))
    for _ in range(samples):
        bits = np.random.randint(0, 2, size=n_qubits)
        label = ''.join(str(b) for b in bits)
        inp = Statevector.from_label(label)
        out = Statevector(target_unitary @ inp.data)
        dataset.append((inp, out))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], QuantumCircuit, ParameterVector, List[Tuple[Statevector, Statevector]], np.ndarray]:
    """Build a variational circuit with shared parameters.

    Parameters
    ----------
    qnn_arch
        Sequence ``[n_qubits, depth]``.
    samples
        Number of training samples.

    Returns
    -------
    arch
        The architecture list.
    circuit
        The fully‑parameterised circuit.
    params
        Parameter vector used in the circuit.
    training_data
        List of (input, target) statepairs.
    target_unitary
        The random unitary that defines the target mapping.
    """
    n_qubits, depth = qnn_arch
    params = ParameterVector("theta", length=depth * n_qubits)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        # Entangle neighbouring qubits
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    target_unitary = _random_unitary(n_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), qc, params, training_data, target_unitary

def feedforward(
    qnn_arch: Sequence[int],
    circuit: QuantumCircuit,
    params: ParameterVector,
    samples: Iterable[Tuple[Statevector, Statevector]]
) -> List[Statevector]:
    """Run the circuit on each input state and return the final statevector."""
    backend = Aer.get_backend("statevector_simulator")
    state_list: List[Statevector] = []
    for inp, _ in samples:
        # Bind random parameters for demonstration
        bound = circuit.bind_parameters({p: np.random.randn() for p in params})
        job = execute(bound, backend, initial_state=inp)
        result = job.result()
        sv = Statevector(result.get_statevector())
        state_list.append(sv)
    return state_list

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure statevectors."""
    return a.fidelity(b)

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    """Construct a graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 2.  Training loop using parameter‑shift rule
# --------------------------------------------------------------------------- #
def train_qnn(
    circuit: QuantumCircuit,
    params: ParameterVector,
    training_data: List[Tuple[Statevector, Statevector]],
    epochs: int = 200,
    lr: float = 0.01
) -> Tuple[List[float], np.ndarray]:
    """Simple training loop for the variational circuit.

    The loss is 1 - fidelity between the circuit output and the target state.
    A finite‑difference approximation of the gradient is used.
    """
    backend = Aer.get_backend("statevector_simulator")
    param_values = np.random.randn(len(params))
    loss_hist: List[float] = []

    for epoch in range(epochs):
        loss = 0.0
        grad = np.zeros_like(param_values)
        for inp, target in training_data:
            # Evaluate at current parameters
            bound = circuit.bind_parameters({p: v for p, v in zip(params, param_values)})
            job = execute(bound, backend, initial_state=inp)
            out = Statevector(job.result().get_statevector())
            fid = out.fidelity(target)
            loss += (1 - fid)

            # Numerical gradient
            eps = 1e-5
            for k in range(len(param_values)):
                plus = param_values.copy()
                minus = param_values.copy()
                plus[k] += eps
                minus[k] -= eps
                bound_plus = circuit.bind_parameters({p: v for p, v in zip(params, plus)})
                bound_minus = circuit.bind_parameters({p: v for p, v in zip(params, minus)})
                job_plus = execute(bound_plus, backend, initial_state=inp)
                job_minus = execute(bound_minus, backend, initial_state=inp)
                out_plus = Statevector(job_plus.result().get_statevector())
                out_minus = Statevector(job_minus.result().get_statevector())
                fid_plus = out_plus.fidelity(target)
                fid_minus = out_minus.fidelity(target)
                grad[k] += -(fid_plus - fid_minus) / (2 * eps)
        loss /= len(training_data)
        grad /= len(training_data)
        param_values -= lr * grad
        loss_hist.append(loss)
    return loss_hist, param_values

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_qnn",
]
