"""ConvGen108: Quantum filter with Qiskit.

This module implements a quantum analogue of the ConvGen108 filter.
Each filter is a parameterised Qiskit circuit with an RY rotation per
pixel, a random entangling block, and measurement of all qubits.
Helper functions for random network generation, feed‑forward
evaluation, and fidelity‑based graph construction mirror the
GraphQNN utilities from the reference pair.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List, Sequence

import networkx as nx
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector


class ConvGen108:
    """Quantum convolution filter using a parameterised Qiskit circuit."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 100,
        threshold: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int, default 2
            Size of the square filter.
        backend : qiskit.providers.BaseBackend, optional
            Backend to execute the circuit on.  If None, AerSimulator is used.
        shots : int, default 100
            Number of shots per execution.
        threshold : float, default 0.0
            Threshold used to decide whether a pixel is encoded as a π rotation or identity.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or AerSimulator()
        # Build circuit
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # add a random two‑qubit entangling layer
        self._circuit += random_circuit(self.n_qubits, depth=2, measure=False)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the filter on a single kernel.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        # compute average probability of |1> per qubit
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt
        return total_ones / (total_counts * self.n_qubits)

    def get_circuit(self) -> QuantumCircuit:
        """Return a transpiled copy of the underlying circuit."""
        return transpile(self._circuit, backend=self.backend)


# --------------------------------------------------------------------------- #
# Helper functions mirroring the GraphQNN utilities
# --------------------------------------------------------------------------- #

def random_training_data(
    unitary: QuantumCircuit,
    samples: int,
    backend: qiskit.providers.BaseBackend | None = None,
    shots: int = 100,
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate synthetic training pairs for a quantum filter.

    Parameters
    ----------
    unitary : QuantumCircuit
        The target unitary circuit.
    samples : int
        Number of samples to generate.
    backend : qiskit.providers.BaseBackend, optional
        Backend to execute the unitary on.  If None, AerSimulator is used.
    shots : int, default 100
        Number of shots per execution.

    Returns
    -------
    List[Tuple[np.ndarray, float]]
        Each tuple contains an input kernel (numpy array) and the
        probability output of the target unitary.
    """
    backend = backend or AerSimulator()
    dataset: List[Tuple[np.ndarray, float]] = []
    for _ in range(samples):
        kernel = np.random.randn(unitary.num_qubits).reshape(
            int(np.sqrt(unitary.num_qubits)),
            int(np.sqrt(unitary.num_qubits)),
        )
        # encode input as rotation angles
        circuit = unitary.copy()
        for i, val in enumerate(kernel.flatten()):
            circuit.ry(val, i)
        circuit.measure_all()
        job = qiskit.execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        total_ones = sum(bitstring.count("1") * cnt for bitstring, cnt in counts.items())
        total_counts = sum(counts.values())
        prob = total_ones / (total_counts * unitary.num_qubits)
        dataset.append((kernel, prob))
    return dataset


def random_network(
    qnn_arch: List[int], samples: int, shots: int = 100
) -> Tuple[List[int], List[ConvGen108], List[Tuple[np.ndarray, float]], ConvGen108]:
    """
    Create a random quantum network of ConvGen108 filters.

    Returns
    -------
    Tuple[List[int], List[ConvGen108], List[Tuple[np.ndarray, float]], ConvGen108]
        Architecture list, list of filter instances, training dataset,
        and the target filter used as ground truth.
    """
    filters: List[ConvGen108] = []
    # For simplicity, all filters share the same kernel size derived from the first layer
    kernel_size = int(np.sqrt(qnn_arch[0]))
    for _ in range(len(qnn_arch) - 1):
        filters.append(ConvGen108(kernel_size=kernel_size))
    target_filter = ConvGen108(kernel_size=int(np.sqrt(qnn_arch[-1])))
    training_data = random_training_data(target_filter.get_circuit(), samples, shots=shots)
    return qnn_arch, filters, training_data, target_filter


def feedforward(
    qnn_arch: List[int],
    filters: List[ConvGen108],
    samples: Iterable[Tuple[np.ndarray, float]],
) -> List[List[float]]:
    """
    Run a sequence of quantum filters on a dataset.

    Parameters
    ----------
    qnn_arch : List[int]
        Architecture.
    filters : List[ConvGen108]
        Filters for each layer.
    samples : Iterable[Tuple[np.ndarray, float]]
        Dataset of inputs and target outputs.

    Returns
    -------
    List[List[float]]
        Output probabilities per sample per layer.
    """
    stored: List[List[float]] = []
    for kernel, _ in samples:
        outputs = [kernel]
        current = kernel
        for filt in filters:
            out = filt.run(current)
            outputs.append(out)
            # broadcast scalar to shape of next filter input
            current = np.full_like(kernel, out)
        stored.append(outputs)
    return stored


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap between two probability vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm) ** 2)


def fidelity_adjacency(
    states: Iterable[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from state fidelities.

    Parameters
    ----------
    states : Iterable[np.ndarray]
        Sequence of probability vectors output by the quantum filters.
    threshold : float
        Fidelity threshold for weight 1 edges.
    secondary : float, optional
        Secondary threshold for weighted edges.
    secondary_weight : float, default 0.5
        Weight for secondary edges.

    Returns
    -------
    networkx.Graph
        Weighted adjacency graph.
    """
    graph = nx.Graph()
    states = list(states)
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(a, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "ConvGen108",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
