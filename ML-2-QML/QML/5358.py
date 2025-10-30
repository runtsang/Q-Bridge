from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def HybridSamplerQNN():
    """Quantum hybrid sampler mirroring the classical counterpart."""
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolution circuit
    def conv_circuit(params):
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

    # Convolution layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = conv_circuit(params[param_index:param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    # Pooling circuit
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    # Pooling layer
    def pool_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = pool_circuit(params[param_index:param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    # Build ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
    ansatz.compose(pool_layer(8, "p1"), [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
    ansatz.compose(conv_layer(4, "c2"), [4, 5, 6, 7], inplace=True)
    ansatz.compose(pool_layer(4, "p2"), [4, 5, 6, 7], inplace=True)
    ansatz.compose(conv_layer(2, "c3"), [6, 7], inplace=True)
    ansatz.compose(pool_layer(2, "p3"), [6, 7], inplace=True)

    # Full circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observables: two PauliZ on the first two qubits
    obs0 = SparsePauliOp.from_list([("Z" * 8, 1)])
    obs1 = SparsePauliOp.from_list([("I" * 7 + "Z", 1)])

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=[obs0, obs1],
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
