"""ConvGen: quantum convolution and graph similarity module.

This module implements a quantum‑inspired convolution filter using a
parameter‑free variational circuit and a graph‑based similarity graph.
It mirrors the classical implementation in the QML counterpart of the
original Conv and GraphQNN utilities.

Author: OpenAI GPT‑OSS‑20B
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qiskit
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen"]


class ConvGen:
    """Unified convolution module with quantum and graph modes."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        mode: str = "quantum",
        backend: str | None = None,
        shots: int = 100,
        quantum_threshold: float = 127,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel.
        threshold : float
            Classical threshold for the sigmoid in the classic branch.
        mode : str
            One of ``'quantum'`` or ``'graph'``.  ``'classic'`` is not
            implemented in the QML module.
        backend : str | None
            Backend name for Qiskit (default: Aer qasm_simulator).
        shots : int
            Number of shots for circuit execution.
        quantum_threshold : float
            Threshold used to map classical input to qubit rotations.
        """
        if mode not in {"quantum", "graph"}:
            raise ValueError(f"Unsupported mode {mode!r}.  Only 'quantum' and 'graph' are available.")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode
        self.shots = shots
        self.quantum_threshold = quantum_threshold

        if mode == "quantum":
            self._setup_quantum(backend)
        elif mode == "graph":
            # graph mode does not need a circuit
            pass

    # --------------------------------------------------------------------
    #  Quantum convolution branch
    # --------------------------------------------------------------------
    def _setup_quantum(self, backend_name: str | None):
        """Create a random two‑qubit‑gate circuit with parameter‑free Rx rotations."""
        self.n_qubits = self.kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        # Parameter‑free Rx rotations
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        # Add a shallow random two‑qubit layer
        self._circuit += random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()

        # Backend
        if backend_name is None:
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
        else:
            self.backend = qiskit.Aer.get_backend(backend_name)

    def _run_quantum(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a 2‑D kernel.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten data to a 1‑D array of qubit values
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in flat:
            bind = {}
            for idx, val in enumerate(row):
                bind[self.theta[idx]] = np.pi if val > self.quantum_threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute mean probability of |1> over all qubits
        total_ones = 0
        total_shots = self.shots * self.n_qubits
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
        return total_ones / total_shots

    # --------------------------------------------------------------------
    #  Graph similarity branch
    # --------------------------------------------------------------------
    def _feature_vector(self, data: np.ndarray) -> np.ndarray:
        """Return a flattened feature vector for the kernel."""
        return data.flatten()

    def run_graph(
        self,
        data_list: list[np.ndarray],
        similarity_threshold: float = 0.8,
        *,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from a list of kernel responses.

        Parameters
        ----------
        data_list : list[np.ndarray]
            List of 2‑D arrays, each of shape (kernel_size, kernel_size).
        similarity_threshold : float
            Primary threshold for edge inclusion.
        secondary_threshold : float | None
            Secondary threshold for weighted edges.
        secondary_weight : float
            Weight assigned to secondary edges.

        Returns
        -------
        networkx.Graph
            Weighted adjacency graph.
        """
        features = [self._feature_vector(d) for d in data_list]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(features)))
        for (i, fi), (j, fj) in itertools.combinations(enumerate(features), 2):
            dot = np.dot(fi, fj)
            norm_i = np.linalg.norm(fi) + 1e-12
            norm_j = np.linalg.norm(fj) + 1e-12
            similarity = dot / (norm_i * norm_j)
            if similarity >= similarity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary_threshold is not None and similarity >= secondary_threshold:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------
    #  Public interface
    # --------------------------------------------------------------------
    def run(self, data: np.ndarray | list[np.ndarray]) -> float | nx.Graph:
        """
        Dispatch based on the selected mode.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            Input data for the selected mode.

        Returns
        -------
        float | nx.Graph
            Result of the operation.
        """
        if self.mode == "quantum":
            if not isinstance(data, np.ndarray):
                raise TypeError("Quantum mode expects a single np.ndarray.")
            return self._run_quantum(data)
        elif self.mode == "graph":
            if not isinstance(data, list):
                raise TypeError("Graph mode expects a list of np.ndarray.")
            return self.run_graph(data)
        else:  # pragma: no cover
            raise RuntimeError("Unsupported mode.")
