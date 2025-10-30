import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN ansatz."""
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Construct a convolutional layer that couples neighboring qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block that reduces entanglement."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Construct a pooling layer mapping source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]), [src, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def QCNN_ansatz() -> QuantumCircuit:
    """Full 8‑qubit QCNN ansatz composed of convolution and pooling layers."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution layer
    ansatz.compose(conv_layer(8, "c1"), inplace=True)

    # First pooling layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second convolution layer
    ansatz.compose(conv_layer(4, "c2"), inplace=True)

    # Second pooling layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Third convolution layer
    ansatz.compose(conv_layer(2, "c3"), inplace=True)

    # Third pooling layer
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit.decompose()

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class EstimatorQNN(FastBaseEstimator):
    """Quantum estimator that wraps the QCNN ansatz and offers a fast evaluation API."""
    def __init__(self) -> None:
        super().__init__(circuit=QCNN_ansatz())

__all__ = ["EstimatorQNN", "FastBaseEstimator"]
