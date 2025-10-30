import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Sequence
import networkx as nx

class QCNNQuantum:
    """Quantum variational QCNN with convolution‑pooling pattern."""

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.circuit = self._build_ansatz()
        self.estimator = StatevectorEstimator()
        self.qnn = self._build_qnn()

    # ---------- Circuit builders ----------
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[idx:idx+3])
            qc.append(sub, [q, q+1])
            idx += 3
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx+3])
            qc.append(sub, [src, snk])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self._conv_layer(8, "c1"), inplace=True)
        qc.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        qc.compose(self._conv_layer(4, "c2"), inplace=True)
        qc.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)
        qc.compose(self._conv_layer(2, "c3"), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return qc

    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZFeatureMap(self.n_qubits)
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(self.circuit, inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        return EstimatorQNN(circuit=circuit.decompose(),
                            observables=observable,
                            input_params=feature_map.parameters,
                            weight_params=self.circuit.parameters,
                            estimator=self.estimator)

    # ---------- Public API ----------
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return model predictions for a batch of classical feature vectors."""
        return self.qnn.forward(data)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        G = nx.Graph()
        n = len(states)
        G.add_nodes_from(range(n))
        for i, a in enumerate(states):
            for j, b in enumerate(states[:i]):
                fid = abs((a.dag() @ b)[0, 0]) ** 2
                if fid >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G
