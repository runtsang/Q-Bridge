from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    for i in range(2):
        circuit.rx(input_params.bs_theta, i)
        circuit.ry(input_params.bs_phi, i)
        circuit.rz(input_params.phases[i], i)
        circuit.rx(_clip(input_params.squeeze_r[i], 5), i)
        circuit.ry(_clip(input_params.displacement_r[i], 5), i)
    for layer in layers:
        for i in range(2):
            circuit.rx(layer.bs_theta, i)
            circuit.ry(layer.bs_phi, i)
            circuit.rz(layer.phases[i], i)
            circuit.rx(_clip(layer.squeeze_r[i], 5), i)
            circuit.ry(_clip(layer.displacement_r[i], 5), i)
    return circuit


def generate_superposition_data(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "generate_superposition_data",
    "build_classifier_circuit",
]
