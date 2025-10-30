"""Quantum circuit that fuses photonic-inspired layers with a QCNN ansatz."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, reused from the original seed."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _photonic_layer_circuit(params: FraudLayerParameters, qubits: list[int], clip: bool = False) -> QuantumCircuit:
    """Build a small photonic-inspired sub‑circuit using standard qiskit gates."""
    circ = QuantumCircuit(*qubits)
    # Beam‑splitter angles -> RZ on the first qubit
    circ.rz(params.bs_theta if not clip else _clip(params.bs_theta, 5), qubits[0])
    circ.rz(params.bs_phi if not clip else _clip(params.bs_phi, 5), qubits[1])
    # Phase shifters
    for i, phase in enumerate(params.phases):
        circ.rz(phase if not clip else _clip(phase, 5), qubits[i])
    # Squeezing → emulate with RY rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circ.ry(r if not clip else _clip(r, 5), qubits[i])
        circ.rz(phi if not clip else _clip(phi, 5), qubits[i])
    # Displacement → additional RZ rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circ.rz(r if not clip else _clip(r, 5), qubits[i])
        circ.ry(phi if not clip else _clip(phi, 5), qubits[i])
    # Kerr non‑linearity → a single RZ gate
    for i, k in enumerate(params.kerr):
        circ.rz(k if not clip else _clip(k, 1), qubits[i])
    return circ


def conv_circuit(sub_params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
    """Small 2‑qubit conv sub‑circuit."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(sub_params[0], 0)
    target.ry(sub_params[1], 1)
    target.cx(0, 1)
    target.ry(sub_params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """QCNN convolutional layer as in the original example."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[param_index:param_index + 3], q1, q2)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = conv_circuit(params[param_index:param_index + 3], q1, q2)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(sub_params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
    """Small 2‑qubit pool sub‑circuit."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(sub_params[0], 0)
    target.ry(sub_params[1], 1)
    target.cx(0, 1)
    target.ry(sub_params[2], 1)
    return target


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """QCNN pooling layer."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        sub = pool_circuit(params[param_index:param_index + 3], source, sink)
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc


def build_hybrid_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Construct a hybrid photonic‑QCNN quantum circuit."""
    # Feature map
    feature_map = ZFeatureMap(8)
    # Base circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)

    # Photonic-inspired layer applied once
    photonic_circ = _photonic_layer_circuit(input_params, list(range(8)), clip=False)
    circuit.append(photonic_circ, range(8))

    # QCNN ansatz
    ansatz = QuantumCircuit(8, name="QCNN Ansatz")
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

    circuit.append(ansatz, range(8))

    # Observable for a single‑qubit measurement on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    return circuit


__all__ = [
    "FraudLayerParameters",
    "build_hybrid_fraud_detection_circuit",
]
