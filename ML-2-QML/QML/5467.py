import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import BaseOperator, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from typing import Sequence, List
from FastBaseEstimator import FastBaseEstimator

class HybridEstimator(FastBaseEstimator):
    """
    Quantum hybrid estimator that supplies a QCNN ansatz, a quanvolution filter,
    and a convenient EstimatorQNN wrapper for end‑to‑end experiments.
    """
    def __init__(self, circuit: QuantumCircuit, observables: Sequence[BaseOperator]) -> None:
        super().__init__(circuit)
        self.circuit = circuit
        self.observables = observables
        self.backend = Aer.get_backend("qasm_simulator")

    # ---------------- QCNN ansatz ------------------------------------
    def conv_circuit(self, params: np.ndarray) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(self, params: np.ndarray) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(self, sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            qc = qc.compose(self.pool_circuit(params[param_index:param_index+3]), [src, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def build_qcnn(self) -> QuantumCircuit:
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self.conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self.pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(self.conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self.pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(self.conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self.pool_layer([0], [1], "p3"), inplace=True)
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit

    # ---------------- Quanvolution filter ----------------------------
    def build_conv_filter(self, kernel_size: int = 2, threshold: float = 127) -> QuantumCircuit:
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(n_qubits, 2)
        qc.measure_all()
        qc.params = theta
        qc.threshold = threshold
        return qc

    # ---------------- EstimatorQNN wrapper --------------------------
    def build_estimator_qnn(self) -> EstimatorQNN:
        circuit = self.build_qcnn()
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        estimator = qiskit.quantum_info.Statevector  # placeholder for actual backend estimator
        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=circuit.parameters,
            weight_params=circuit.parameters,
            estimator=estimator,
        )

    # ---------------- Evaluate QCNN ----------------------------------
    def evaluate_qcnn(self, data: np.ndarray) -> List[complex]:
        """
        Evaluate the QCNN ansatz on classical data using the EstimatorQNN.
        """
        qnn = self.build_estimator_qnn()
        param_vector = ParameterVector("x", length=data.shape[1])
        feature_map = ZFeatureMap(data.shape[1])
        bound = feature_map.assign_parameters({param_vector[i]: val for i, val in enumerate(data[0])})
        full_circuit = bound.compose(qnn.circuit, inplace=False)
        job = execute(full_circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(full_circuit)
        exp = 0.0
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            exp += (1 if bitstring[-1] == '0' else -1) * cnt
        return exp / total

__all__ = ["HybridEstimator"]
