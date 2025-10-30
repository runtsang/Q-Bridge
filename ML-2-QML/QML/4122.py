"""
QuantumGraphFusion: Quantum interface of the hybrid module.

This module builds a parameterised quantum circuit that mirrors the
classical fully‑connected + sampler network.  It receives a vector of
parameters (thetas), constructs a 2‑node adjacency matrix using a
deterministic rule (e.g., thresholding the first parameter), and then
runs a variational circuit that applies Ry(theta) gates and CNOTs
according to the adjacency.  The expectation value of the first qubit
is returned.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Tuple

__all__ = ["QuantumGraphFusion"]


class QuantumGraphFusion:
    """
    Quantum interface that implements a variational circuit based on
    a sampled adjacency matrix.
    """

    def __init__(
        self,
        graph_nodes: int = 2,
        backend: str | None = None,
        shots: int = 1024,
    ) -> None:
        self.graph_nodes = graph_nodes
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit_cache: dict[Tuple[int,...], QuantumCircuit] = {}

    def _adjacency_from_thetas(self, thetas: np.ndarray) -> np.ndarray:
        """
        Use QSamplerQNN to produce a 2‑node edge probability from the
        first two thetas.  The remaining weights are sampled randomly.
        """
        # Build the sampler circuit from the reference
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        sampler_qnn = QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

        # Bind the first two thetas as inputs.  Randomly initialise weights.
        param_dict = {
            inputs[0]: thetas[0],
            inputs[1]: thetas[1],
        }
        for w in weights:
            param_dict[w] = np.random.randn()

        probs = sampler_qnn.predict(param_dict)  # shape (4,)
        # Probability of the "11" state (both qubits 1) is taken as edge weight
        edge_prob = probs[3]

        adj = np.eye(self.graph_nodes, dtype=float)
        adj[0, 1] = adj[1, 0] = edge_prob
        return adj

    def _build_circuit(self, adj: np.ndarray, thetas: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that applies Ry(theta_i) to each qubit and a
        CNOT for each edge indicated by the adjacency matrix.
        """
        n = adj.shape[0]
        qc = QuantumCircuit(n)
        # Apply Ry gates with the given parameters
        for i in range(n):
            qc.ry(thetas[i], i)
        # Apply CNOTs for edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0.5:
                    qc.cx(i, j)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation value of
        the first qubit.
        """
        thetas = np.asarray(thetas, dtype=float)
        adj = self._adjacency_from_thetas(thetas)
        key = tuple(adj.ravel())
        if key not in self._circuit_cache:
            qc = self._build_circuit(adj, thetas)
            self._circuit_cache[key] = qc
        else:
            qc = self._circuit_cache[key]

        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        exp = sum(
            (1 if bitstring[-1] == "1" else -1) * cnt for bitstring, cnt in counts.items()
        ) / self.shots
        return np.array([exp])
