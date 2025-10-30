"""Hybrid quantum classifier that stitches a QCNN ansatz with a variational classifier."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used by the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-3.1415926535 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(3.1415926535 / 2, 0)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Build a convolutional layer by pairing qubits and applying the 2‑qubit unit."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], (qubits[2::2] + [0])):
        qc.append(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit – identical to the convolution unit but without the final rotation."""
    qc = QuantumCircuit(2)
    qc.rz(-3.1415926535 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps a set of source qubits onto a smaller set of sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_index : param_index + 3]), [src, snk])
        qc.barrier()
        param_index += 3
    return qc


def _qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
    """Full QCNN ansatz consisting of alternating convolution and pooling layers."""
    qc = QuantumCircuit(num_qubits)

    # First convolution (full size)
    qc.append(_conv_layer(num_qubits, "c1"), list(range(num_qubits)))

    # First pooling (halve)
    qc.append(_pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"),
              list(range(num_qubits)))

    # Second convolution (half size)
    half = num_qubits // 2
    qc.append(_conv_layer(half, "c2"), list(range(half, num_qubits)))

    # Second pooling (quarter size)
    qc.append(_pool_layer(list(range(half // 2)), list(range(half // 2, half)), "p2"),
              list(range(half, num_qubits)))

    # Third convolution (quarter size)
    quarter = half // 2
    qc.append(_conv_layer(quarter, "c3"), list(range(half + quarter, num_qubits)))

    # Third pooling (final reduction)
    qc.append(_pool_layer([half + quarter], [half + quarter + 1], "p3"),
              list(range(half + quarter, num_qubits)))

    return qc


def build_classifier_circuit(
    num_qubits: int, depth: int, use_qcnn: bool = False, qcnn_qubits: int = 8
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz that optionally starts with a QCNN block and ends with a variational classifier.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits available for the circuit.
    depth : int
        Number of variational layers applied after the QCNN block.
    use_qcnn : bool
        Whether to prepend the QCNN ansatz.
    qcnn_qubits : int
        Size of the QCNN sub‑circuit (must be <= num_qubits).

    Returns
    -------
    circuit : QuantumCircuit
        The assembled quantum circuit.
    encoding : list
        Parameters encoding the classical data (feature map).
    weights : list
        Parameters controlling the variational classifier.
    observables : list
        Pauli observables used as measurement operators.
    """
    # Feature map
    feature_map = ZFeatureMap(num_qubits)
    encoding = list(feature_map.parameters)

    # Start with feature encoding
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, list(range(num_qubits)), inplace=True)

    # Optional QCNN block
    if use_qcnn:
        qc_qcnn = _qcnn_ansatz(qcnn_qubits)
        circuit.compose(qc_qcnn, list(range(qcnn_qubits)), inplace=True)

    # Variational classifier layers
    weight_params = ParameterVector("theta", num_qubits * depth)
    weight_index = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weight_params[weight_index], q)
            weight_index += 1
        # Entangle all qubits pairwise with CZ
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Observables – a single Z on the last qubit per output class
    observables = [SparsePauliOp("I" * (num_qubits - 1) + "Z")]

    return circuit, encoding, weight_params, observables


__all__ = ["build_classifier_circuit"]
