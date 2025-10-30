"""Hybrid quantum estimator mirroring the classical structure with Qiskit and Strawberry Fields."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, Any

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


class QuantumSelfAttention:
    """Quantum self‑attention block based on a parameterised circuit."""

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> Dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


class HybridEstimator:
    """Quantum estimator combining self‑attention, a SamplerQNN, and a fraud‑detection program."""

    def __init__(self) -> None:
        # Self‑attention component
        self.attention = QuantumSelfAttention(n_qubits=4)

        # SamplerQNN component
        self.sampler = self._create_sampler_qnn()

        # Fraud‑detection program
        input_params = FraudLayerParameters(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        layers = [input_params]  # single layer for demonstration
        self.fraud_program = build_fraud_detection_program(input_params, layers)

    def _create_sampler_qnn(self) -> QiskitSamplerQNN:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def evaluate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all components with the supplied parameters.

        Expected keys in ``params``:
            - ``attention``: dict with ``rotation_params`` and ``entangle_params`` (np.ndarray)
            - ``sampler``: dict with ``input_params`` and ``weight_params`` (np.ndarray)
            - ``fraud``: dict with ``parameter_sets`` (list of FraudLayerParameters)
        """
        results: Dict[str, Any] = {}

        # Self‑attention
        att_params = params.get("attention", {})
        if att_params:
            results["attention"] = self.attention.run(
                att_params["rotation_params"],
                att_params["entangle_params"],
                shots=att_params.get("shots", 1024)
            )

        # SamplerQNN
        samp_params = params.get("sampler", {})
        if samp_params:
            results["sampler"] = self.sampler.sample(
                samp_params["input_params"],
                samp_params["weight_params"]
            )

        # Fraud detection
        fraud_params = params.get("fraud", {})
        if fraud_params:
            # For demonstration we compute the statevector after applying the program
            state = Statevector.from_instruction(self.fraud_program)
            results["fraud_statevector"] = state

        return results


__all__ = ["HybridEstimator", "FraudLayerParameters", "build_fraud_detection_program"]
