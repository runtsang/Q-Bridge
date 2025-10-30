import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

class QuantumClassifierModel:
    """Quantum counterpart of the hybrid model.  It constructs a variational
    circuit that emulates a QCNN, a graph‑based state‑propagation layer,
    and a sampler for probability outputs."""
    def __init__(self, num_qubits: int = 8, depth: int = 3,
                 graph: nx.Graph | None = None, sampler: bool = True) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.graph = graph or nx.complete_graph(num_qubits)
        self.sampler_enabled = sampler

        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.observables = [SparsePauliOp("Z" + "I" * (num_qubits - 1))]
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self._get_ansatz_params(),
            estimator=self.estimator,
        )
        if self.sampler_enabled:
            self.sampler = StatevectorSampler()
            self.sampler_qnn = SamplerQNN(
                circuit=self.circuit.decompose(),
                input_params=self.feature_map.parameters,
                weight_params=self._get_ansatz_params(),
                sampler=self.sampler,
            )

    def _get_ansatz_params(self) -> ParameterVector:
        return ParameterVector("theta", self.num_qubits * self.depth)

    def _build_data_encoding(self) -> QuantumCircuit:
        enc = QuantumCircuit(self.num_qubits)
        for q, param in enumerate(self.feature_map.parameters):
            enc.ry(param, q)
        return enc

    def _build_conv_layer(self, qubits: List[int], prefix: str) -> QuantumCircuit:
        layer = QuantumCircuit(self.num_qubits)
        params = ParameterVector(prefix, 3 * len(qubits) // 2)
        idx = 0
        for i in range(0, len(qubits) - 1, 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(params[idx], 0)
            sub.ry(params[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(params[idx + 2], 1)
            sub.cx(1, 0)
            sub.rz(np.pi / 2, 0)
            layer.append(sub.to_instruction(), [qubits[i], qubits[i + 1]])
            layer.barrier()
            idx += 3
        return layer

    def _build_pool_layer(self, qubits: List[int], prefix: str) -> QuantumCircuit:
        layer = QuantumCircuit(self.num_qubits)
        params = ParameterVector(prefix, 3 * len(qubits) // 2)
        idx = 0
        for i in range(0, len(qubits) - 1, 2):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(params[idx], 0)
            sub.ry(params[idx + 1], 1)
            sub.cx(0, 1)
            sub.ry(params[idx + 2], 1)
            layer.append(sub.to_instruction(), [qubits[i], qubits[i + 1]])
            layer.barrier()
            idx += 3
        return layer

    def _build_graph_layer(self) -> QuantumCircuit:
        layer = QuantumCircuit(self.num_qubits)
        for u, v in self.graph.edges():
            layer.cx(u, v)
        return layer

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.compose(self._build_data_encoding(), inplace=True)
        for d in range(self.depth):
            ansatz.compose(self._build_conv_layer(list(range(self.num_qubits)), f"c{d}"), inplace=True)
            ansatz.compose(self._build_pool_layer(list(range(self.num_qubits)), f"p{d}"), inplace=True)
            ansatz.compose(self._build_graph_layer(), inplace=True)
        return ansatz

    def forward(self, inputs: List[float]) -> List[float]:
        param_dict = dict(zip(self.feature_map.parameters, inputs))
        return self.qnn.evaluate(param_dict).values

    def sample(self, inputs: List[float], shots: int = 1024) -> dict[int, int]:
        if not self.sampler_enabled:
            raise RuntimeError("Sampler not enabled during construction.")
        param_dict = dict(zip(self.feature_map.parameters, inputs))
        return self.sampler_qnn.sample(param_dict, shots=shots).sample_counts

__all__ = ["QuantumClassifierModel"]
