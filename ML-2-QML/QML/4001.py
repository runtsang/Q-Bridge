"""Quantum circuit implementation of the depth‑wise QCNN ansatz.

This module exposes a function that returns an EstimatorQNN instance,
ready to be used as a layer in a hybrid neural network.  The circuit
mirrors the classical convolutional structure from the first reference
while incorporating quantum feature mapping and parameterised layers.
"""

from __future__ import annotations

import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-qiskit.circuit.library.ConstantGate(-qiskit.math.pi / 2), 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(qiskit.circuit.library.ConstantGate(qiskit.math.pi / 2), 0)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_circuit(params[i*3:(i+1)*3])
        qc.append(block, [i, i+1])
        qc.barrier()
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-qiskit.circuit.library.ConstantGate(-qiskit.math.pi / 2), 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, sink, val in zip(sources, sinks, params):
        block = _pool_circuit(ParameterVector(val.name, length=3))
        qc.append(block, [src, sink])
        qc.barrier()
    return qc

def QCNNQuantumCircuit(n_qubits: int = 8, backend=None, shots: int = 512) -> EstimatorQNN:
    """Builds and returns an EstimatorQNN representing the QCNN ansatz."""
    algorithm_globals.random_seed = 12345
    backend = backend or qiskit.Aer.get_backend("aer_simulator")
    estimator = Estimator()
    feature_map = ZFeatureMap(n_qubits)
    feature_map.decompose()

    # Construct ansatz layers
    ansatz = QuantumCircuit(n_qubits)
    ansatz.compose(_conv_layer(n_qubits, "c1"))
    ansatz.compose(_pool_layer(list(range(n_qubits)), list(range(n_qubits, 2*n_qubits)), "p1"))
    ansatz.compose(_conv_layer(n_qubits, "c2"))
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p2"))
    ansatz.compose(_conv_layer(2, "c3"))
    ansatz.compose(_pool_layer([0], [1], "p3"))

    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNQuantumCircuit"]
