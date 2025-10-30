"""HybridSamplerQNN: quantum sampler inspired by SamplerQNN and FraudDetection.

The circuit consists of a stack of parameterized layers. Each layer contains
RY rotations on the two qubits (input parameters), a parameterised RZZ
entangling gate (weight parameter), and additional single‑qubit RZ rotations
to emulate the phase and displacement controls from the photonic model.
The final statevector is sampled with a StatevectorSampler and wrapped in
Qiskit Machine Learning's SamplerQNN for easy integration with classical
optimisers.

The design mirrors the classical fraud‑style layers by providing a
parameterised block that can be repeated, allowing the model to learn
complex correlations while keeping the parameter count scalable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RZZGate, RZGate
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


@dataclass
class QuantumLayerParameters:
    """Parameters for a single quantum layer."""
    input_angles: Tuple[float, float]
    weight_angle: float
    rz_angles: Tuple[float, float]


def _quantum_layer(circuit: QuantumCircuit, params: QuantumLayerParameters, qubits: Tuple[int, int]) -> None:
    """Append a fraud‑style quantum block to the circuit."""
    q0, q1 = qubits
    circuit.ry(params.input_angles[0], q0)
    circuit.ry(params.input_angles[1], q1)
    circuit.append(RZZGate(params.weight_angle), [q0, q1])
    circuit.append(RZGate(params.rz_angles[0]), [q0])
    circuit.append(RZGate(params.rz_angles[1]), [q1])


def build_hybrid_sampler_qnn(
    input_params: QuantumLayerParameters,
    layer_params: Iterable[QuantumLayerParameters],
) -> QiskitSamplerQNN:
    """Construct a Qiskit SamplerQNN mirroring the classical fraud‑style layers."""
    num_layers = 1 + len(layer_params)
    input_vector = ParameterVector("input", 2 * num_layers)
    weight_vector = ParameterVector("weight", 1 * num_layers)
    rz_vector = ParameterVector("rz", 2 * num_layers)

    qc = QuantumCircuit(2)
    qubits = (0, 1)

    # Build layers
    for i, params in enumerate([input_params] + list(layer_params)):
        idx = i
        layer = QuantumLayerParameters(
            input_angles=(input_vector[2 * idx], input_vector[2 * idx + 1]),
            weight_angle=weight_vector[idx],
            rz_angles=(rz_vector[2 * idx], rz_vector[2 * idx + 1]),
        )
        _quantum_layer(qc, layer, qubits)

    qc.measure_all()

    sampler = StatevectorSampler()
    qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=input_vector,
        weight_params=weight_vector,
        sampler=sampler,
    )
    return qnn
