"""Quantum component of the hybrid QCNN: variational convolution & pooling layers."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary."""
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolution block over all qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling between specified source‑sink pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        sub = pool_circuit(params[idx : idx + 3])
        qc.append(sub, [src, snk])
        qc.barrier()
        idx += 3
    return qc


def QCNN_QML(num_qubits: int = 8, feature_dim: int = 8) -> EstimatorQNN:
    """Builds the full QCNN variational circuit and returns an EstimatorQNN."""
    estimator = Estimator()
    feature_map = ZFeatureMap(feature_dim)

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First convolution & pooling
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(
        pool_layer(
            list(range(num_qubits // 2)),
            list(range(num_qubits // 2, num_qubits)),
            "p1",
        ),
        inplace=True,
    )

    # Second convolution & pooling
    ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
    ansatz.compose(
        pool_layer(
            list(range(num_qubits // 4)),
            list(range(num_qubits // 4, num_qubits // 2)),
            "p2",
        ),
        inplace=True,
    )

    # Third convolution & pooling
    ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )


# Optional fully‑connected quantum layer example (FCL)
class QuantumFCL:
    """Parameterised quantum circuit emulating a fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, shots: int = 100):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.measure_all()
        self.theta = theta

    def run(self, thetas: np.ndarray) -> np.ndarray:
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = job.result().get_counts(self.circuit)
        probs = np.array(list(result.values())) / self.shots
        states = np.array([int(k, 2) for k in result.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])
