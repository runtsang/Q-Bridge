from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import networkx as nx
from typing import List, Sequence


class QCNNGen227:
    """
    Quantum implementation of a QCNN ansatz with configurable depth and dynamic pooling.
    A Z‑feature map encodes classical data; the ansatz consists of alternating
    convolution and pooling layers.  The class also exposes utilities to compute
    state‑fidelity and to construct a fidelity‑based adjacency graph from a set
    of input samples.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.estimator = StatevectorEstimator()
        self.obs = SparsePauliOp.from_list([("I" * num_qubits, 1)])

        # Separate weight parameters (those not in the feature map)
        self.weight_params = [
            p for p in self.circuit.parameters if p not in self.feature_map.parameters
        ]

    # ------------------------------------------------------------------ #
    # QCNN building blocks
    # ------------------------------------------------------------------ #
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolution")
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        for q1, q2 in zip(range(1, num_qubits, 2), list(range(2, num_qubits, 2)) + [0]):
            sub = self._conv_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(
        self, sources: List[int], sinks: List[int], prefix: str
    ) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx : idx + 3])
            qc.append(sub, [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # First convolution
        qc.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        # First pooling
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 2)), list(range(self.num_qubits // 2, self.num_qubits)), "p1"
            ),
            inplace=True,
        )
        # Subsequent layers
        current_qubits = self.num_qubits // 2
        for d in range(2, self.depth + 1):
            qc.compose(self._conv_layer(current_qubits, f"c{d}"), inplace=True)
            qc.compose(
                self._pool_layer(
                    list(range(current_qubits // 2)),
                    list(range(current_qubits // 2, current_qubits)),
                    f"p{d}",
                ),
                inplace=True,
            )
            current_qubits //= 2
        return qc

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    def evaluate(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Evaluate the circuit on a batch of classical inputs.
        Each input must be a flat array of length ``self.num_qubits``.
        Returns a list of statevectors.
        """
        param_dict = {}
        for inp in inputs:
            for p, v in zip(self.feature_map.parameters, inp):
                param_dict[p] = v
        # Initialise all weight parameters to zero
        param_dict.update({p: 0.0 for p in self.weight_params})
        results = self.estimator.run(self.circuit, param_dict, observables=[self.obs])
        return [r.statevector for r in results]

    def state_fidelity(self, sv1: np.ndarray, sv2: np.ndarray) -> float:
        """
        Compute the squared absolute overlap between two statevectors.
        """
        return float(np.abs(np.vdot(sv1, sv2)) ** 2)

    def fidelity_graph(
        self,
        inputs: Sequence[np.ndarray],
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted adjacency graph from the statevectors obtained
        for the supplied inputs.
        """
        states = self.evaluate(inputs)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, sv_i in enumerate(states):
            for j in range(i + 1, len(states)):
                sv_j = states[j]
                fid = self.state_fidelity(sv_i, sv_j)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["QCNNGen227"]
