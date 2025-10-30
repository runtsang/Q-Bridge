"""Hybrid quantum‑classical QCNN with kernel estimation and fidelity graphs.

The :class:`QCNNHybrid` class below builds a Qiskit circuit that
combines:

* a ZFeatureMap for data encoding,
* a hierarchical QCNN ansatz (convolution + pooling layers),
* a simple variational kernel circuit used to compute overlaps
  between classical data embeddings,
* a graph of state fidelities that can be used to analyse
  the expressivity of the circuit.

An :class:`EstimatorQNN` is instantiated for fast inference,
and a lightweight :class:`FastBaseEstimator` is provided for
expectation‑value evaluation of arbitrary observables.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import networkx as nx
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ------------------------------------------------------------------
# Lightweight estimator for quantum circuits
# ------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    The implementation follows the reference FastBaseEstimator but
    is self‑contained for this module.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


# ------------------------------------------------------------------
# QCNNHybrid quantum class
# ------------------------------------------------------------------
class QCNNHybrid:
    """Quantum implementation of the hybrid QCNN.

    Parameters
    ----------
    n_qubits
        Number of qubits used for the feature map and the ansatz.
    seed
        Random seed for reproducibility.
    """
    def __init__(self, n_qubits: int = 8, seed: int = 12345) -> None:
        self.n_qubits = n_qubits
        self.random_state = np.random.RandomState(seed)

        # Feature map
        self.feature_map = ZFeatureMap(n_qubits)

        # Kernel ansatz (simple ry rotations)
        self.kernel_ansatz = self._build_kernel_ansatz()

        # QCNN ansatz
        self.cnn_ansatz = self._build_qcnn_ansatz()

        # Assemble full circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        self.circuit.compose(self.cnn_ansatz, range(n_qubits), inplace=True)

        # Observable (measure first qubit in Z)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        # Estimator
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.cnn_ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _build_kernel_ansatz(self) -> QuantumCircuit:
        """Return a simple parameterised kernel circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(0.0, i)  # placeholder; parameters will be bound later
        return qc

    def _build_qcnn_ansatz(self) -> QuantumCircuit:
        """Construct the hierarchical QCNN ansatz from the reference."""
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, qubits)
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, range(num_qubits))
            return qc

        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(conv_layer(self.n_qubits, "c1"), list(range(self.n_qubits)), inplace=True)

        # First Pooling Layer
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(self.n_qubits)), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(conv_layer(4, "c2"), list(range(4, self.n_qubits)), inplace=True)

        # Second Pooling Layer
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, self.n_qubits)), inplace=True)

        # Third Convolutional Layer
        ansatz.compose(conv_layer(2, "c3"), list(range(6, self.n_qubits)), inplace=True)

        # Third Pooling Layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, self.n_qubits)), inplace=True)

        return ansatz

    # ------------------------------------------------------------------
    # Kernel and graph utilities
    # ------------------------------------------------------------------
    def kernel_matrix(self, batch1: Sequence[torch.Tensor], batch2: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the quantum kernel matrix between two batches."""
        a = torch.stack(batch1).detach().cpu().numpy()
        b = torch.stack(batch2).detach().cpu().numpy()
        mat = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            state_x = Statevector.from_instruction(self._encode_circuit(x))
            for j, y in enumerate(b):
                state_y = Statevector.from_instruction(self._encode_circuit(y))
                mat[i, j] = abs(state_x.overlap(state_y)) ** 2
        return mat

    def _encode_circuit(self, vec: np.ndarray) -> QuantumCircuit:
        """Return a circuit that encodes a single classical vector with ry gates."""
        qc = QuantumCircuit(self.n_qubits)
        for idx, val in enumerate(vec):
            qc.ry(val, idx)
        return qc

    def fidelity_graph(self, states: Sequence[Statevector], threshold: float) -> nx.Graph:
        """Build a graph based on state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fid = abs(states[i].overlap(states[j])) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=fid)
        return graph

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Delegate to :class:`FastBaseEstimator`."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------
    # Call operator
    # ------------------------------------------------------------------
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the QCNN for a batch of inputs."""
        return self.qnn(inputs)

__all__ = ["QCNNHybrid", "FastBaseEstimator"]
