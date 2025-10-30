"""Quantum implementation of a QCNN‑style circuit that incorporates the
quantum filter from Conv.py as a sub‑circuit.

The design follows the structure of the classical QCNN helper while
adding a QuanvCircuit for each qubit to emulate the original
quantum convolution filter.  The resulting EstimatorQNN can be trained
with a variational optimiser on a simulator or a real device.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# Quantum filter from Conv.py
class QuanvCircuit:
    """Sub‑circuit implementing the quantum filter used in Conv.py."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def get_instruction(self) -> qiskit.circuit.Instruction:
        return self._circuit.to_instruction()

# Quantum convolution and pooling primitives from QCNN seed
def conv_circuit_original(params):
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

def conv_circuit(params, theta_params):
    """Conv layer that first applies a QuanvCircuit to each qubit
    and then the original two‑qubit conv circuit.
    """
    qc = QuantumCircuit(2)
    # Apply QuanvCircuit to each qubit with its own theta
    for i in range(2):
        qc.compose(QuanvCircuit(1, Aer.get_backend('qasm_simulator'), 100, 127).get_instruction(), [i], inplace=True)
    # Append the original conv circuit
    qc.compose(conv_circuit_original(params), [0, 1], inplace=True)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3], None), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index+3], None), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def HybridConvQCNN() -> EstimatorQNN:
    """Builds a QCNN‑style EstimatorQNN that uses the QuanvCircuit
    as part of each convolutional layer.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridConvQCNN"]
