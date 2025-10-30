"""Hybrid fraud detection model – quantum component.

This module builds a photonic feature extractor using Strawberry Fields
and a Qiskit‑based sampler that classifies the extracted features.
The resulting class, `FraudDetectionHybrid`, can be used to generate
probabilities from quantum circuits and, with parameter binding, can
participate in joint optimisation schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


# --------------------------------------------------------------------------- #
# 1. Photonic layer definition – same as the classical seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """
    Create a Strawberry Fields program that mirrors the layered
    photonic architecture.  The first layer is unclipped; subsequent
    layers are clipped to keep amplitudes bounded.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


# --------------------------------------------------------------------------- #
# 2. Qiskit sampler circuit – quantum analogue of the classical SamplerQNN
# --------------------------------------------------------------------------- #
def _build_sampler_circuit() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Construct a parameterised 2‑qubit sampler circuit.
    Returns the circuit, the input parameters, and the weight parameters.
    """
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
    qc.measure_all()
    return qc, inputs, weights


def _build_qiskit_sampler() -> QiskitSamplerQNN:
    """
    Wrap the Qiskit sampler circuit in the Qiskit Machine Learning SamplerQNN
    interface.  The sampler uses the Aer qasm simulator by default.
    """
    qc, inputs, weights = _build_sampler_circuit()
    sampler = StatevectorSampler()
    return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)


# --------------------------------------------------------------------------- #
# 3. Hybrid quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Quantum‑centric fraud detection model.

    The class builds a photonic feature extractor and a Qiskit sampler.
    It exposes a `classify` method that accepts a 2‑element input vector
    (e.g., raw transaction features) and returns a probability of fraud.
    """

    def __init__(self, input_params: FraudLayerParameters, hidden_params: Sequence[FraudLayerParameters]) -> None:
        # Photonic feature extractor
        self.photonic_program = build_fraud_detection_program(input_params, hidden_params)

        # Qiskit sampler for classification
        self.sampler = _build_qiskit_sampler()

    def classify(self, feature_vector: list[float]) -> dict[str, float]:
        """
        Run the photonic circuit (simulated) and the Qiskit sampler to obtain
        fraud probabilities.

        Parameters
        ----------
        feature_vector : list[float]
            Two‑dimensional input to be fed into the sampler's input parameters.

        Returns
        -------
        dict
            Mapping from class labels to probabilities.
        """
        # Bind input parameters for the sampler
        bind_dict = {var: val for var, val in zip(self.sampler.input_params, feature_vector)}

        # Execute the sampler; the result is a dict of counts
        result_counts = self.sampler.run(bindings=bind_dict).get_counts()

        # Normalise to probabilities
        total = sum(result_counts.values())
        probs = {label: count / total for label, count in result_counts.items()}

        # Map binary counts to a single fraud probability (e.g., label '1')
        fraud_prob = probs.get("1", 0.0) + probs.get("01", 0.0)  # adjust as needed
        return {"fraud_probability": fraud_prob}

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "FraudDetectionHybrid"]
