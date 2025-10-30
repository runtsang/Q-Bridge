"""Quantum QCNN with adaptive depth and noise‑aware measurement."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[idx : idx + 3])
        qc.append(sub, [i, i + 1])
        idx += 3
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = _pool_circuit(params[idx : idx + 3])
        qc.append(sub, [i, i + 1])
        idx += 3
    return qc

def QCNNEnhanced(num_qubits: int = 8, depth: int = 3) -> EstimatorQNN:
    """Return a variational QCNN with adaptive depth and noise‑aware measurement."""
    # Feature map
    feature_map = ZFeatureMap(num_qubits, reps=1, entanglement="full")
    # Ansatz construction
    ansatz = QuantumCircuit(num_qubits)
    current_qubits = num_qubits
    for d in range(depth):
        ansatz.compose(_conv_layer(current_qubits, f"c{d}"), inplace=True)
        ansatz.compose(_pool_layer(current_qubits // 2, f"p{d}"), inplace=True)
        current_qubits //= 2
    # Combine feature map and ansatz
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    # Observable
    observable = SparsePauliOp.from_list([("Z" + "I" * (circuit.num_qubits - 1), 1)])
    # Noise model: simple depolarizing error on all gates
    noise = NoiseModel()
    depol = depolarizing_error(0.01, 1)
    noise.add_all_qubit_quantum_error(depol, "u1")
    noise.add_all_qubit_quantum_error(depol, "u2")
    noise.add_all_qubit_quantum_error(depol, "cx")
    # Estimator with noise
    simulator = AerSimulator(noise_model=noise)
    estimator = Estimator(backend=simulator, seed=12345)
    # QNN
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNEnhanced"]
