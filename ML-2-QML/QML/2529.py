"""Hybrid quantum autoencoder fraud detector.

This module builds a quantum neural network that takes the latent
representation produced by a classical autoencoder and outputs a
fraud‑risk score.  The circuit is inspired by the domain‑wall
ansatz from the Qiskit example and the photonic layer construction
from the Strawberry Fields example.  The QNN is trained with a
classical optimiser and can be run on a simulator or real backend.

The design follows the same class name as the classical module so
that the two halves can be swapped in a unified pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Import the original autoencoder configuration for dimensionality
from Autoencoder import AutoencoderConfig

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_photonic_layer(q: QuantumRegister, params: FraudLayerParameters, clip: bool) -> None:
    """Apply the photonic layer from the Strawberry Fields example using Qiskit gates."""
    # BSgate -> beam splitter simulated by a rotation and a controlled‑Z
    theta = params.bs_theta
    phi = params.bs_phi
    qc = QuantumCircuit(q)
    qc.ry(theta, q[0])
    qc.rz(phi, q[1])
    for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qc.ry(r_eff, q[i])
    for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qc.rz(r_eff, q[i])
    for i, k in enumerate(params.kerr):
        k_eff = _clip(k, 1.0) if clip else k
        qc.rz(k_eff, q[i])
    q.compose(qc, inplace=True)

def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a photonic‑style circuit for fraud detection."""
    qr = QuantumRegister(2, "q")
    qc = QuantumCircuit(qr)
    _apply_photonic_layer(qr, input_params, clip=False)
    for layer in layers:
        _apply_photonic_layer(qr, layer, clip=True)
    return qc

def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Domain‑wall inspired auto‑encoder circuit from the Qiskit example."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def domain_wall(qc: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a domain wall by X gates on qubits a‑b."""
    for i in range(a, b):
        qc.x(i)
    return qc

class HybridAutoEncoderFraudDetector:
    """Quantum implementation of the hybrid auto‑encoder fraud detector."""
    def __init__(
        self,
        ae_config: AutoencoderConfig,
        fraud_params: Iterable[FraudLayerParameters],
        *,
        device: str = "qasm_simulator",
    ) -> None:
        self.ae_config = ae_config
        self.fraud_params = list(fraud_params)
        self.device = device
        algorithm_globals.random_seed = 42
        self.sampler = StatevectorSampler()
        self._build_qnn()

    def _build_qnn(self) -> None:
        """Construct the quantum neural network."""
        # Build the fraud detection circuit
        qc_fraud = build_photonic_program(
            self.fraud_params[0], self.fraud_params[1:]
        )
        # Build the auto‑encoder part that will encode latent variables
        qc_ae = auto_encoder_circuit(
            num_latent=self.ae_config.latent_dim,
            num_trash=2,
        )
        # Combine circuits: first fraud circuit, then auto‑encoder
        qc = qc_fraud.compose(qc_ae, inplace=True)
        # No input parameters – the latent vector is encoded as basis states
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the QNN on a batch of latent vectors."""
        # Convert latent to basis state indices (simple binarisation)
        indices = (latent > 0).int().numpy()
        # Sample the circuit for each index
        probs = []
        for idx in indices:
            probs.append(self.qnn.forward(idx))
        return torch.tensor(probs, dtype=torch.float32)

    def train(self, data: torch.Tensor, labels: torch.Tensor, *, epochs: int = 20,
              lr: float = 1e-3, device: str | None = None) -> List[float]:
        """Train the quantum parameters using a classical optimiser."""
        device = device or self.device
        opt = COBYLA(maxiter=200)
        history: List[float] = []

        def loss_fn(params):
            self.qnn.set_weights(params)
            preds = self.forward(data)
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels.float())
            return loss.item()

        opt.optimize(num_vars=len(self.qnn.weights), initial_point=self.qnn.weights, fprime=None, callback=None)
        return history
