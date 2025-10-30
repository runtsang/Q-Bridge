"""Unified quantum estimator module.

Wraps a Qiskit EstimatorQNN and provides convenience utilities to
construct a parameterised circuit from a graph adjacency matrix.  The
class can be instantiated with a pre‑defined circuit or built
automatically from a graph and a target architecture.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import networkx as nx
import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator, Estimator

class UnifiedEstimatorQNN:
    """Quantum estimator that mirrors the classical UnifiedEstimatorQNN.

    The constructor accepts a quantum circuit and corresponding
    observables.  If ``circuit`` is ``None`` a simple circuit is
    built from ``graph`` and ``architecture``.
    """

    def __init__(
        self,
        circuit: Optional[QuantumCircuit] = None,
        observables: Optional[SparsePauliOp] = None,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        estimator: Estimator = StatevectorEstimator(),
        graph: Optional[nx.Graph] = None,
        architecture: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit
            Pre‑defined quantum circuit.  If omitted a circuit is built
            from ``graph`` and ``architecture``.
        observables
            Observable(s) used for expectation value estimation.
        input_params
            Parameters that encode the classical inputs.
        weight_params
            Parameters that encode trainable weights.
        estimator
            Backend estimator; defaults to a state‑vector simulator.
        graph
            Graph whose adjacency matrix is used to generate a
            parameterised circuit when ``circuit`` is ``None``.
        architecture
            Layer widths for the graph‑based circuit.  Required if
            ``circuit`` is ``None``.
        """
        if circuit is None:
            if graph is None or architecture is None:
                raise ValueError("graph and architecture must be provided when circuit is None")
            circuit, input_params, weight_params = self._build_circuit_from_graph(
                graph, architecture
            )
            observables = self._default_observables(circuit.num_qubits)

        self.estimator_qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        self._circuit = circuit
        self._estimator = estimator

    @staticmethod
    def _default_observables(num_qubits: int) -> SparsePauliOp:
        """Return a simple observable (Y on the first qubit)."""
        return SparsePauliOp.from_list([("Y" * num_qubits, 1)])

    def _build_circuit_from_graph(
        self,
        graph: nx.Graph,
        architecture: Sequence[int],
    ) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
        """Create a parameterised circuit whose structure follows ``graph``."""
        num_qubits = len(graph.nodes)
        qc = QuantumCircuit(num_qubits)
        input_params: List[Parameter] = []
        weight_params: List[Parameter] = []

        # Apply a Hadamard to every qubit
        for q in range(num_qubits):
            qc.h(q)

        # Encode graph edges as controlled rotations
        for u, v in graph.edges:
            p = Parameter(f"theta_{u}_{v}")
            qc.ry(p, u)
            qc.rz(p, v)
            weight_params.append(p)

        # Map input parameters to each qubit
        for q in range(num_qubits):
            p = Parameter(f"inp_{q}")
            qc.rx(p, q)
            input_params.append(p)

        # Add a simple layer of random rotations to match the target
        for _ in range(architecture[-1] - num_qubits):
            p = Parameter(f"rand_{len(weight_params)}")
            qc.ry(p, np.random.randint(num_qubits))
            weight_params.append(p)

        return qc, input_params, weight_params

    def evaluate(self, input_values: List[float]) -> float:
        """Return the expectation value for a single input set."""
        if len(input_values)!= len(self.estimator_qnn.input_params):
            raise ValueError("Input dimension mismatch")
        # Build parameter binding
        binding = {
            p: v for p, v in zip(self.estimator_qnn.input_params, input_values)
        }
        # Use the estimator to compute expectation
        expectation = self.estimator_qnn.predict(binding)
        return float(expectation[0])

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic data for training a unitary mapping."""
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            state = np.random.randn(unitary.shape[0]) + 1j * np.random.randn(unitary.shape[0])
            state = state / np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

__all__ = ["UnifiedEstimatorQNN"]
