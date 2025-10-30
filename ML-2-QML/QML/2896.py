"""Quantum implementation of the QCNN with convolution and pooling blocks."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from typing import List

__all__ = ["QCNNQuantum", "QuantumFCL"]


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution unitary used in all convolutional layers.
    """
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


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Builds a convolutional layer that applies _conv_circuit to each
    pair of adjacent qubits and then to a shifted pair.
    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[idx : idx + 3]), [q1, q2])
        qc.barrier()
        idx += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[idx : idx + 3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling unitary used in all pooling layers.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """
    Builds a pooling layer that maps a set of source qubits to a smaller set of sinks.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        qc.append(_pool_circuit(params[idx : idx + 3]), [src, snk])
        qc.barrier()
        idx += 3
    return qc


def QCNNQuantum() -> EstimatorQNN:
    """
    Construct the full quantum QCNN as an EstimatorQNN.

    The circuit comprises:
        1. A Z‑feature map on 8 qubits.
        2. Three convolution–pooling pairs with decreasing qubit counts.
        3. A Pauli‑Z observable on the first qubit.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution–pooling pair
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second convolution–pooling pair
    ansatz.compose(_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolution–pooling pair
    ansatz.compose(_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Full circuit
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


def QuantumFCL(n_qubits: int = 1, shots: int = 1024):
    """
    Simple fully‑connected quantum layer that returns expectation values
    of a single Pauli‑Z measurement after applying a parameterised Ry gate.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 1).
    shots : int
        Number of shots for the simulator (default 1024).

    Returns
    -------
    circuit : qiskit.quantum_info.Statevector
        A circuit object that can be executed with a backend.
    """
    from qiskit import Aer, execute
    from qiskit.circuit import Parameter

    backend = Aer.get_backend("qasm_simulator")
    qc = QuantumCircuit(n_qubits)
    theta = Parameter("θ")
    qc.h(range(n_qubits))
    qc.ry(theta, range(n_qubits))
    qc.measure_all()

    def run(thetas: List[float]) -> np.ndarray:
        job = execute(
            qc,
            backend,
            shots=shots,
            parameter_binds=[{theta: th} for th in thetas],
        )
        result = job.result().get_counts(qc)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    return run
