import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.providers.aer import AerSimulator


class _HybridQCNN:
    """Quantum implementation of the hybrid QCNN architecture."""
    def __init__(self):
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self._observable(),
            input_params=self._feature_map().parameters,
            weight_params=self._ansatz().parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # circuit building blocks
    # ------------------------------------------------------------------
    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolution")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        param_index = 0
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[param_index:param_index + 3])
            qc.append(sub, [i, i + 1])
            param_index += 3
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling")
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        param_index = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index:param_index + 3])
            qc.append(sub, [src, sink])
            param_index += 3
        return qc

    # ------------------------------------------------------------------
    # higher‑level components
    # ------------------------------------------------------------------
    def _feature_map(self):
        return ZFeatureMap(8)

    def _ansatz(self):
        ansatz = QuantumCircuit(8, name="HybridAnsatz")
        ansatz.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return ansatz

    def _observable(self):
        return SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def _build_circuit(self):
        circuit = QuantumCircuit(8)
        circuit.compose(self._feature_map(), range(8), inplace=True)
        circuit.compose(self._ansatz(), range(8), inplace=True)
        return circuit

    # ------------------------------------------------------------------
    # utility functions
    # ------------------------------------------------------------------
    def fidelity_adjacency(self,
                           states,
                           threshold,
                           secondary=None,
                           secondary_weight=0.5):
        """Build weighted graph from pairwise state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fid = np.abs((states[i].data @ states[j].data.conj().T)[0, 0]) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def quantum_kernel_matrix(self, a, b):
        """Compute kernel matrix via state overlap using RX encodings."""
        def encode(vec):
            qc = QuantumCircuit(4)
            for i, val in enumerate(vec):
                qc.rx(val, i)
            return qc

        sim = AerSimulator(method="statevector")
        mat = np.zeros((len(a), len(b)), dtype=float)
        for i, ca in enumerate(a):
            job_a = sim.run(encode(ca))
            sv_a = job_a.result().get_statevector()
            for j, cb in enumerate(b):
                job_b = sim.run(encode(cb))
                sv_b = job_b.result().get_statevector()
                mat[i, j] = np.abs(np.vdot(sv_a, sv_b)) ** 2
        return mat


def QCNNHybridQNN() -> EstimatorQNN:
    """Return a ready‑to‑use EstimatorQNN implementing the hybrid QCNN."""
    return _HybridQCNN().qnn


__all__ = ["QCNNHybridQNN", "_HybridQCNN"]
