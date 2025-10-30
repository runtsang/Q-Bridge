"""Quantum circuit factory for a hybrid QCNN‑style ansatz.

The circuit combines a ZFeatureMap data‑encoding with a stack of
parameterised convolution and pooling blocks.  The depth argument
controls how many conv/pool pairs are inserted, allowing a direct
comparison with the classical depth‑controlled network above.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution kernel used by all conv layers."""
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
    """Construct a convolutional layer that applies the 2‑qubit kernel
    to all adjacent pairs in a round‑robin fashion."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    param_index = 0
    qubits = list(range(num_qubits))
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling kernel – identical to the conv kernel but
    without the final RZ(π/2) on qubit 0."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Apply the pooling kernel to each source–sink pair."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    param_index = 0
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index+3]), [src, snk])
        qc.barrier()
        param_index += 3
    return qc

def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a QCNN‑style quantum circuit and its metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map (must match the data
        dimensionality).
    depth : int
        Number of convolution / pooling pairs to insert.  A depth of 1
        produces a single conv/pool pair, depth 2 stacks two pairs, etc.

    Returns
    -------
    circuit : QuantumCircuit
        Decomposed QCNN circuit ready for use with EstimatorQNN.
    encoding : list[ParameterVector]
        Feature‑map parameters.
    weights : list[ParameterVector]
        Ansatz parameters.
    observables : list[SparsePauliOp]
        Measurement operators – a single Z on the first qubit is
        sufficient for binary classification.
    """
    # Data‑encoding layer
    feature_map = ZFeatureMap(num_qubits)
    circuit = QuantumCircuit(num_qubits)

    # Build the ansatz by stacking conv/pool pairs
    ansatz = QuantumCircuit(num_qubits)
    for i in range(depth):
        # Convolution
        ansatz.append(conv_layer(num_qubits, f"c{i+1}"), range(num_qubits))
        # Pooling – we keep all qubits but only measure the first.
        ansatz.append(pool_layer(list(range(num_qubits)), list(range(num_qubits)), f"p{i+1}"),
                      range(num_qubits))

    # Combine feature map and ansatz
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: single Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    # Decompose to avoid nested instructions
    circuit = circuit.decompose()

    # Extract parameter vectors
    encoding = list(feature_map.parameters)
    weights = [p for p in ansatz.parameters if p not in encoding]

    return circuit, encoding, weights, [observable]

__all__ = ["build_classifier_circuit"]
