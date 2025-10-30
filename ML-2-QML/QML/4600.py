"""Quantum hybrid estimator that evaluates a parameterised circuit and can compute fidelity‑based graphs.

The implementation follows the spirit of the original ``FastBaseEstimator`` seed but is
entirely quantum: it accepts a ``qiskit.QuantumCircuit`` (or a circuit returned by
``UnifiedEstimator.FCL``) and evaluates it either via state‑vector simulation or a
QASM simulator with shot noise.  Additionally, helper functions for building random
quantum neural networks and for constructing fidelity‑based adjacency graphs are
provided, mirroring the GraphQNN utilities from the reference seeds.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import Parameter
from typing import Iterable, Sequence, List, Optional, Any
import networkx as nx
import qutip as qt
import scipy as sc
import itertools

class UnifiedEstimator:
    """Quantum estimator that evaluates a parameterised circuit and can compute fidelity graphs."""

    def __init__(self, circuit: QuantumCircuit, shots: int = 1024, backend: Optional[Any] = None) -> None:
        self.circuit = circuit
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            if self.shots > 0:
                job = execute(circ, self.backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts(circ)
                probs = np.array(list(counts.values())) / self.shots
                outcomes = np.array([int(k, 2) for k in counts.keys()]).astype(float)
                exp = (outcomes * probs).sum()
                row = [exp for _ in observables]
            else:
                sv = Statevector.from_instruction(circ)
                row = [sv.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, si in enumerate(states):
            for j, sj in enumerate(states[i + 1 :], i + 1):
                fid = abs((si.dag() * sj)[0, 0]) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def FCL(n_qubits: int = 1, shots: int = 1024) -> "UnifiedEstimator":
        """Return a simple quantum fully‑connected layer circuit."""
        theta = Parameter("theta")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.ry(theta, range(n_qubits))
        qc.measure_all()
        return UnifiedEstimator(qc, shots=shots)

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[tuple[qt.Qobj, qt.Qobj]] = []
        dim = int(np.log2(unitary.dim[0]))
        for _ in range(samples):
            state = qt.rand_ket(dim)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = UnifiedEstimator._random_qubit_unitary(qnn_arch[-1])
        training_data = UnifiedEstimator.random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            nin = qnn_arch[layer - 1]
            nout = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out in range(nout):
                op = UnifiedEstimator._random_qubit_unitary(nin + 1)
                if nout > 1:
                    op = qt.tensor(UnifiedEstimator._random_qubit_unitary(nin + 1), UnifiedEstimator._tensored_id(nout - 1))
                    op = UnifiedEstimator._swap_registers(op, nin, nin + out)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        return qt.qeye(2 ** num_qubits)

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(op.dims[0][0]))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)
