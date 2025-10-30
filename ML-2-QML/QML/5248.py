from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.primitives import Estimator
from qiskit import Aer
from typing import Iterable, Tuple, List, Sequence

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def conv_circuit(params: Sequence[ParameterVector]) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the QCNN ansatz."""
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
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: Sequence[ParameterVector]) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def build_qcnn_ansatz() -> QuantumCircuit:
    """Constructs the full QCNN ansatz used for fraud detection."""
    circuit = QuantumCircuit(8)
    # First Convolutional Layer
    circuit.compose(conv_layer(8, "c1"), range(8), inplace=True)
    # First Pooling Layer
    circuit.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    # Second Convolutional Layer
    circuit.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    # Second Pooling Layer
    circuit.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    # Third Convolutional Layer
    circuit.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    # Third Pooling Layer
    circuit.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
    return circuit

def build_fraud_detection_circuit() -> QuantumCircuit:
    """Combine a feature map with the QCNN ansatz into a single circuit."""
    feature_map = ZFeatureMap(8)
    ansatz = build_qcnn_ansatz()
    full_circuit = QuantumCircuit(8)
    full_circuit.compose(feature_map, range(8), inplace=True)
    full_circuit.compose(ansatz, range(8), inplace=True)
    return full_circuit

class FraudDetectionHybrid:
    """Quantum circuit wrapper that evaluates a QCNN‑based fraud detection model.
    The circuit accepts two sets of parameters:
        - feature_map parameters (input) of length 8
        - ansatz parameters (weights) of length equal to the number of free parameters
    The result is a two‑class probability vector obtained by sampling the first qubit."""
    def __init__(self, backend=None, shots: int = 1024) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = build_fraud_detection_circuit()
        # Count free parameters for binding
        self.input_params = [p for p in self.circuit.parameters if "theta" in p.name]
        self.weight_params = [p for p in self.circuit.parameters if "c" in p.name or "p" in p.name]

    def run(self, input_values: np.ndarray, weight_values: np.ndarray) -> np.ndarray:
        """Execute the circuit with the supplied parameters and return a probability pair."""
        param_dict = {p: v for p, v in zip(self.input_params + self.weight_params, np.concatenate([input_values, weight_values]))}
        bound_circuit = self.circuit.bind_parameters(param_dict)
        transpiled = transpile(bound_circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Probability of measuring '0' on the first qubit (assuming 8‑bit basis strings)
        prob0 = sum(counts.get(f"0{bit:07b}", 0) for bit in range(128)) / self.shots
        prob1 = 1.0 - prob0
        return np.array([prob0, prob1])

__all__ = ["FraudDetectionHybrid", "build_fraud_detection_circuit"]
