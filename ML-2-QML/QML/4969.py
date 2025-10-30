"""Hybrid quantum fraud detection module.

Provides a photonic fraud‑detection program and a Qiskit variational circuit
that implements self‑attention, a lightweight estimator and a classification ansatz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitStatevectorEstimator

# Seed imports
from FraudDetection import FraudLayerParameters
from QuantumClassifierModel import build_classifier_circuit as build_q_classifier

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_fraud_layer(modes: Sequence, params: FraudLayerParameters, clip: bool) -> None:
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

def build_photonic_fraud_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_fraud_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_fraud_layer(q, layer, clip=True)
    return program

def build_quantum_attention_classifier(
    num_qubits: int,
    depth: int,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    # Encoding
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for p, q in zip(encoding, range(num_qubits)):
        circuit.rx(p, q)

    # Variational block
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Self‑attention sub‑circuit
    for q in range(num_qubits):
        circuit.rx(rotation_params[3 * q], q)
        circuit.ry(rotation_params[3 * q + 1], q)
        circuit.rz(rotation_params[3 * q + 2], q)
    for q in range(num_qubits - 1):
        circuit.crx(entangle_params[q], q, q + 1)

    # Observables for estimator
    obs = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), obs

class FraudDetectionHybrid:
    """
    Quantum hybrid fraud‑detection module that bundles a photonic program
    and a Qiskit variational circuit with self‑attention, estimator and classifier.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        attention_params: np.ndarray,
        num_qubits: int = 4,
        depth: int = 2,
    ) -> None:
        self.fraud_params = fraud_params
        self.attention_params = attention_params
        self.num_qubits = num_qubits
        self.depth = depth

        # Photonic fraud‑detection program
        self.photonic_prog = build_photonic_fraud_program(fraud_params, [])

        # Qiskit variational + attention + classifier circuit
        self.quantum_circuit, self.enc_params, self.var_params, self.observables = \
            build_quantum_attention_classifier(
                num_qubits=num_qubits,
                depth=depth,
                rotation_params=attention_params,
                entangle_params=np.zeros(num_qubits - 1),
            )

        # EstimatorQNN wrapper (state‑vector backend)
        self.estimator = QiskitEstimatorQNN(
            circuit=self.quantum_circuit,
            observables=self.observables,
            input_params=self.enc_params,
            weight_params=self.var_params,
            estimator=QiskitStatevectorEstimator(),
        )

    def get_photonic_program(self) -> sf.Program:
        """Return the photonic fraud‑detection program."""
        return self.photonic_prog

    def get_quantum_circuit(self) -> QuantumCircuit:
        """Return the Qiskit variational circuit with attention."""
        return self.quantum_circuit

    def run_estimator(self, inputs: np.ndarray) -> np.ndarray:
        """Run the estimator on given classical inputs."""
        return self.estimator.run(inputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        End‑to‑end forward pass: photonic simulation → quantum estimator → classification logits.
        """
        # Photonic simulation (state‑vector only)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        result = eng.run(self.photonic_prog, args={"inputs": inputs})
        # Simplified feature extraction: photon number expectation of first mode
        photon_counts = np.array([np.real(result.state.expectation_value(sf.ops.Dgate(1, 0).to_matrix()))])
        # Run quantum estimator
        est_out = self.estimator.run(photon_counts)
        return est_out

__all__ = ["FraudDetectionHybrid"]
