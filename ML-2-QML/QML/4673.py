"""QCNNGen119: Quantum convolution‑pool‑sampler network."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, ParameterVector, transpile, assemble
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper: Convolution & pooling sub‑circuits
# --------------------------------------------------------------------------- #
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


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3 // 2)
    idx = 0
    for q in range(0, num_qubits, 2):
        sub = _conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q, q + 1])
        idx += 3
    return qc


def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q in range(0, num_qubits, 2):
        sub = _pool_circuit(params[idx : idx + 3])
        qc.append(sub, [q, q + 1])
        idx += 3
    return qc


# --------------------------------------------------------------------------- #
# Main QCNN ansatz with a Sampler block
# --------------------------------------------------------------------------- #
def QCNNGen119QNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz: 3 conv‑pool stages
    ansatz = QuantumCircuit(8, name="QCNN-Ansatz")
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # Append sampler (2‑qubit) to the end of the circuit
    sampler_qc = QuantumCircuit(2, name="Sampler")
    sampler_qc.ry(ParameterVector("s1", 1)[0], 0)
    sampler_qc.cx(0, 1)
    sampler_qc.ry(ParameterVector("s2", 1)[0], 1)
    sampler = SamplerQNN(circuit=sampler_qc, input_params=ParameterVector("s1", 1),
                        weight_params=ParameterVector("s2", 1), sampler=StatevectorSampler())

    # Combine feature map, ansatz and sampler
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    # The sampler operates on the last two qubits of the ansatz
    circuit.compose(sampler_qc, [6, 7], inplace=True)

    observable = qiskit.quantum_info.SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters + sampler_qc.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNGen119QNN"]
