"""Quantum self‑attention with fraud‑detection inspired post‑processing.

The quantum component implements a variational rotation‑entanglement
circuit that outputs a probability distribution over the qubits.
These probabilities are interpreted as attention weights and used to
produce a weighted sum of the classical inputs.  The resulting vector
is then passed through a small classical fraud‑detection stack, mirroring
the photonic architecture.

Key features
------------
* The circuit is built from rotation angles (rx, ry, rz) and controlled‑rx
  entanglement gates, matching the classical interface.
* The output is a single‑dimensional regression value.
* The module is fully compatible with the original ``SelfAttention`` factory.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterable, Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
# Parameter container – same as the classical version
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Classical fraud‑detection block (used after the quantum attention)
# --------------------------------------------------------------------------- #
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> torch.nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = torch.nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = torch.nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# --------------------------------------------------------------------------- #
# Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumFraudSelfAttention:
    """Hybrid quantum‑classical self‑attention model."""
    def __init__(self, n_qubits: int, fraud_params: Sequence[FraudLayerParameters]):
        self.n_qubits = n_qubits
        self.fraud_params = fraud_params
        self.backend = AerSimulator()

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        # Apply rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        probs = {int(k, 2): v / shots for k, v in counts.items()}
        # Attention weights over qubits
        weights = torch.tensor([probs.get(i, 0.0) for i in range(self.n_qubits)], dtype=torch.float32)
        # Weighted sum of classical inputs
        weighted = torch.matmul(weights.unsqueeze(0), torch.as_tensor(inputs, dtype=torch.float32))
        # Classical fraud‑detection post‑processing
        x = weighted
        for layer in [_layer_from_params(p, clip=True) for p in self.fraud_params]:
            x = layer(x)
        x = torch.nn.Linear(2, 1)(x)
        return x.detach().numpy()

# --------------------------------------------------------------------------- #
# Factory function matching the original interface
# --------------------------------------------------------------------------- #
def SelfAttention() -> QuantumFraudSelfAttention:
    dummy_params = [
        FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.5,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
    ]
    return QuantumFraudSelfAttention(n_qubits=4, fraud_params=dummy_params)

__all__ = ["SelfAttention", "FraudLayerParameters", "QuantumFraudSelfAttention"]
