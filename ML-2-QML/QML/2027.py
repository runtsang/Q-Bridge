"""
Module: qcnn_enhanced_qml
Implements an extended quantum convolutional neural network with adaptive pooling,
multi‑observable measurement, and noise‑aware simulation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQNN
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator, Unroller
from qiskit.quantum_info import Pauli
from qiskit.extensions import depolarizing_error
from qiskit.circuit.library.standard_gates import IGate
from typing import List

# ----------------------------------------------------------------------
# Helper circuits
# ----------------------------------------------------------------------


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Base convolution unitary acting on two qubits.
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Pooling unitary that merges two qubits into one, leaving a single logical qubit.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int,
               param_prefix: str,
               layer_name: str = "Convolutional Layer") -> QuantumCircuit:
    """
    Compose a convolutional layer over adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits, name=layer_name)
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
        subc = conv_circuit(params[i * 3:(i + 1) * 3])
        qc.append(subc, [q1, q2])
        qc.barrier()
    return qc


def pool_layer(sources: List[int],
               sinks: List[int],
               param_prefix: str,
               layer_name: str = "Pooling Layer") -> QuantumCircuit:
    """
    Compose a pooling layer that maps each source qubit into a sink qubit.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name=layer_name)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        subc = pool_circuit(params[i * 3:(i + 1) * 3])
        qc.append(subc, [src, snk])
        qc.barrier()
    return qc


# ----------------------------------------------------------------------
# Main QNN construction
# ----------------------------------------------------------------------


def QCNNEnhanced() -> EstimatorQNN:
    """
    Build a QCNN with the following features:
    * Three convolutional + pooling stages.
    * Adaptive pooling: outputs from multiple qubits are measured.
    * Multi‑observable measurement for richer feature extraction.
    * Depolarising noise added to each two‑qubit gate for realistic simulation.
    """
    # Set global seed for reproducibility
    np.random.seed(12345)

    # Feature map
    feature_map = ZFeatureMap(8)
    feature_map_params = feature_map.parameters

    # Ansätze
    ansatz = QuantumCircuit(8, name="Ansatz")

    # 1st conv layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # 1st pool layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # 2nd conv layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # 2nd pool layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # 3rd conv layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # 3rd pool layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map & ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Add depolarising noise to every two‑qubit gate
    noise_model = AerSimulator().noise_model
    for gate_name, gate in circuit.gates:
        if gate.num_qubits == 2:
            noise_model.add_quantum_error(depolarizing_error(0.01, 2), gate_name)

    # Observables: measure Z on each qubit (multi‑observable)
    observables = [SparsePauliOp.from_list([(f"Z{'I'*i}{'I'*(7-i)}", 1)]) for i in range(8)]

    # Build QNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map_params,
        weight_params=ansatz.parameters,
        estimator=AerSimulator(noise_model=noise_model),
    )
    return qnn


__all__ = ["QCNNEnhanced"]
