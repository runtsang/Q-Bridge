"""GraphQNN: Quantum implementation using Qiskit.

Features
--------
* Parameterised quantum circuits per GNN layer.
* Random generation of unitary layers via Haar‑distributed matrices.
* Feed‑forward propagation producing statevectors for each graph node.
* Fidelity‑based adjacency graph construction from pure state overlaps.
* Estimator based on Qiskit ``Statevector`` for expectation values.

The quantum API mirrors the classical one while exposing a quantum‑centric
interface.  It is intentionally lightweight to run on the local Aer simulator.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import scipy as sc

from.FastBaseEstimator import FastBaseEstimator


def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a Haar‑random unitary matrix."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    q, _ = sc.linalg.qr(mat)
    return q


def random_training_data(unitary: np.ndarray, samples: int) -> List[tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic data by applying a target unitary to random states."""
    dataset: List[tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        vec /= sc.linalg.norm(vec)
        target = unitary @ vec
        dataset.append((vec, target))
    return dataset


class GraphQNNQuantum:
    """Quantum graph neural network using Qiskit parameterised circuits.

    Parameters
    ----------
    qnn_arch : list[int]
        Number of qubits per layer.  Each layer corresponds to a unitary
        acting on ``layer[i] + 1`` qubits (input + output).
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.arch = list(qnn_arch)
        self.circuits: List[QuantumCircuit] = []

        for inp_q, out_q in zip(qnn_arch[:-1], qnn_arch[1:]):
            qc = QuantumCircuit(inp_q + out_q)
            theta = Parameter("theta")
            # Simple variational ansatz: a layer of Ry rotations
            for q in range(inp_q + out_q):
                qc.ry(theta, q)
            self.circuits.append(qc)

    def _bind_circuit(self, circuit: QuantumCircuit, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= 1:
            raise ValueError("Each circuit expects a single parameter ``theta``.")
        return circuit.assign_parameters({circuits[0].parameters[0]: params[0]}, inplace=False)

    def feedforward(
        self,
        samples: Iterable[np.ndarray],
    ) -> List[List[np.ndarray]]:
        """Apply the per‑layer circuits to each input state."""
        stored: List[List[np.ndarray]] = []
        backend = Aer.get_backend("statevector_simulator")

        for state in samples:
            layerwise = [state]
            current = state
            for qc in self.circuits:
                # bind a single random parameter for demonstration
                bound = qc.assign_parameters({qc.parameters[0]: np.random.rand()}, inplace=False)
                job = execute(bound, backend)
                sv = Statevector(job.result().get_statevector(bound))
                current = sv.data
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from overlaps of pure quantum states."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = abs(np.vdot(s_i, s_j)) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def evaluate(
        self,
        observables: Iterable[np.ndarray],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        estimator = FastBaseEstimator(self._circuit_wrapper)
        return estimator.evaluate(observables, parameter_sets)

    def _circuit_wrapper(self, params: Sequence[float]) -> QuantumCircuit:
        # Create a single circuit by concatenating all layers with the same theta
        qc = QuantumCircuit(sum(self.arch))
        for i, layer_qc in enumerate(self.circuits):
            theta = params[i] if i < len(params) else params[-1]
            bound = layer_qc.assign_parameters({layer_qc.parameters[0]: theta}, inplace=False)
            qc.append(bound, qc.qubits)
        return qc


__all__ = [
    "GraphQNNQuantum",
    "random_training_data",
    "fidelity_adjacency",
]
