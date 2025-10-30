import numpy as np
from typing import Iterable, Sequence, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class FastEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        from qiskit.quantum_info import Statevector
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._statevector = Statevector

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = self._statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class QCNNHybrid:
    """Quantum QCNN circuit with convolution and pooling layers."""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)
        self.circuit = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        self.fast_estimator = FastEstimator(self.circuit)

    def conv_circuit(self, params):
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

    def conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(self, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self.pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.feature_map, inplace=True)
        # First Convolutional Layer
        qc.append(self.conv_layer(self.n_qubits, "c1"), list(range(self.n_qubits)))
        # First Pooling Layer
        qc.append(self.pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(self.n_qubits)))
        # Second Convolutional Layer on remaining qubits
        qc.append(self.conv_layer(self.n_qubits // 2, "c2"), list(range(self.n_qubits // 2)))
        # Second Pooling Layer
        qc.append(self.pool_layer([0, 1], [2, 3], "p2"), list(range(self.n_qubits // 2)))
        # Third Convolutional Layer on remaining qubits
        qc.append(self.conv_layer(self.n_qubits // 4, "c3"), list(range(self.n_qubits // 4)))
        # Third Pooling Layer
        qc.append(self.pool_layer([0], [1], "p3"), list(range(2)))
        return qc

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute output probabilities for a batch of classical input vectors.
        Each input vector is first encoded via the feature map; the remaining
        parameters are random initial weights for the ansatz.
        """
        param_sets: List[List[float]] = []
        for inp in inputs:
            if len(inp)!= len(self.feature_map.parameters):
                raise ValueError("Input vector length must match feature map parameters.")
            weights = np.random.rand(len(self.circuit.parameters) - len(self.feature_map.parameters))
            param_sets.append(list(inp) + list(weights))
        results = self.fast_estimator.evaluate([self.observable], param_sets)
        probs = np.array([float(r[0]) for r in results])
        return probs

__all__ = ["QCNNHybrid"]
