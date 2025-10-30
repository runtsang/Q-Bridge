"""Hybrid quantum circuit for graph‑based QCNN experiments.

This module extends the original QCNN implementation with a graph‑aware
convolution layer.  The conv_layer operates on qubit pairs that are
connected in a supplied adjacency matrix, allowing the quantum circuit
to respect the underlying graph structure.  The module also offers
classical utilities for generating random training data, fidelity‑based
adjacency graphs, and a lightweight `run_qcnn` helper that executes
the circuit on a quantum simulator.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. QCNN primitives
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used by all conv layers."""
    qc = QuantumCircuit(2, name="conv_circuit")
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary used by all pool layers."""
    qc = QuantumCircuit(2, name="pool_circuit")
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


# --------------------------------------------------------------------------- #
# 2. Graph‑aware QCNN construction
# --------------------------------------------------------------------------- #

def create_qcnn_circuit(
    graph: nx.Graph,
    feature_map: QuantumCircuit | None = None,
    num_layers: int = 3,
) -> QuantumCircuit:
    """Build a QCNN circuit that respects the topology of `graph`.

    Parameters
    ----------
    graph : nx.Graph
        Graph whose nodes are mapped to qubits.
    feature_map : QuantumCircuit, optional
        Feature‑map circuit applied to every qubit before the ansatz.
    num_layers : int
        Number of convolution–pooling pairs.

    Returns
    -------
    QuantumCircuit
        Fully‑formed QCNN circuit ready for simulation or execution.
    """
    num_qubits = graph.number_of_nodes()
    qc = QuantumCircuit(num_qubits, name="QCNN")

    # 1. Feature map
    if feature_map is not None:
        for q in range(num_qubits):
            qc.append(feature_map, [q])

    # 2. Convolution + pooling stages
    for l in range(num_layers):
        # Convolution: apply to every edge
        for i, j in graph.edges():
            conv = _conv_circuit(ParameterVector(f"c{l}_{i}_{j}", length=3))
            qc.append(conv, [i, j])

        # Pooling: pair each node with a neighbour to reduce qubit count
        if num_qubits > 1:
            sinks = list(range(1, num_qubits, 2))
            sources = list(range(0, num_qubits, 2))
            for src, sink in zip(sources, sinks):
                pool = _pool_circuit(ParameterVector(f"p{l}_{src}_{sink}", length=3))
                qc.append(pool, [src, sink])

    return qc


def bind_qcnn_parameters(
    circuit: QuantumCircuit,
    param_values: Dict[str, float],
) -> QuantumCircuit:
    """Bind a dictionary of parameter names to numerical values."""
    return circuit.bind_parameters(param_values)


# --------------------------------------------------------------------------- #
# 3. Training data generation
# --------------------------------------------------------------------------- #

def random_training_data_qnn(num_qubits: int, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate random input–output state pairs using a random unitary."""
    dim = 2 ** num_qubits
    rng = np.random.default_rng()
    # Random unitary via QR decomposition
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(a)
    U = Operator(q)
    dataset = []
    for _ in range(samples):
        psi_vec = rng.standard_normal((dim,)) + 1j * rng.standard_normal((dim,))
        psi_vec /= np.linalg.norm(psi_vec)
        psi = Statevector(psi_vec)
        dataset.append((psi, U @ psi))
    return dataset


def random_network_qnn(num_qubits: int, samples: int):
    """Return a random unitary and a training dataset derived from it."""
    dim = 2 ** num_qubits
    rng = np.random.default_rng()
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(a)
    U = Operator(q)
    training_data = random_training_data_qnn(num_qubits, samples)
    return U, training_data


# --------------------------------------------------------------------------- #
# 4. Forward propagation
# --------------------------------------------------------------------------- #

def run_qcnn(
    circuit: QuantumCircuit,
    param_values: Dict[str, float],
    input_state: Statevector,
) -> Statevector:
    """Execute the QCNN circuit on a single input state."""
    bound_circuit = bind_qcnn_parameters(circuit, param_values)
    backend = AerSimulator(method="statevector")
    job = backend.run(bound_circuit, initial_state=input_state)
    return Statevector(job.result().get_statevector())


def feedforward_qnn(
    circuit: QuantumCircuit,
    param_values: Dict[str, float],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[Tuple[Statevector, Statevector]]:
    """Run the QCNN circuit on each input state and collect the outputs."""
    results = []
    for inp, tgt in samples:
        out = run_qcnn(circuit, param_values, inp)
        results.append((out, tgt))
    return results


# --------------------------------------------------------------------------- #
# 5. Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity_qnn(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure states."""
    return float(abs(np.vdot(a.data, b.data)) ** 2)


def fidelity_adjacency_qnn(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_qnn(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "create_qcnn_circuit",
    "bind_qcnn_parameters",
    "random_training_data_qnn",
    "random_network_qnn",
    "run_qcnn",
    "feedforward_qnn",
    "state_fidelity_qnn",
    "fidelity_adjacency_qnn",
]
