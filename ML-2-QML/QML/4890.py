"""Hybrid quantum autoencoder that fuses a feature‑map, RealAmplitudes ansatz and a fraud‑inspired gate block.

The quantum encoder maps classical input into a quantum state via a parameterised feature map,
then applies a variational RealAmplitudes ansatz.  A fraud‑layer block injects additional
non‑linear transformations before measurement.  The resulting latent vector is decoded
classically by a linear layer.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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


def _apply_fraud_layer(circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Append a photonic‑style gate block to the circuit."""
    # Simple mapping of photonic gates to rotation gates
    circuit.ry(params.bs_theta if not clip else _clip(params.bs_theta, 5.0), 0)
    circuit.rz(params.bs_phi if not clip else _clip(params.bs_phi, 5.0), 0)
    for i, phase in enumerate(params.phases):
        circuit.rz(phase if not clip else _clip(phase, 5.0), i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.ry(r if not clip else _clip(r, 5.0), i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rz(r if not clip else _clip(r, 5.0), i)
    for i, k in enumerate(params.kerr):
        circuit.rx(k if not clip else _clip(k, 1.0), i)


def _build_quantum_circuit(
    num_qubits: int,
    fraud_params: FraudLayerParameters,
    fraud_layers: Iterable[FraudLayerParameters],
) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """Construct the full quantum encoder circuit."""
    input_params: List[Parameter] = [Parameter(f"x{i}") for i in range(num_qubits)]
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)

    # Feature‑map: encode input data into RY rotations
    for i, p in enumerate(input_params):
        circuit.ry(p, i)

    # Variational ansatz
    ansatz = RealAmplitudes(num_qubits, reps=2)
    circuit.compose(ansatz, inplace=True)

    # Fraud‑layer block (only applied to first qubit for brevity)
    _apply_fraud_layer(circuit, fraud_params, clip=True)

    # Measurement of all qubits
    circuit.measure_all()
    return circuit, input_params, ansatz.parameters


class HybridAutoencoder(nn.Module):
    """Quantum‑classical hybrid autoencoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters] = (),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        circuit, self.input_params, weight_params = _build_quantum_circuit(
            latent_dim, fraud_params, fraud_layers
        )
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.input_params,
            weight_params=weight_params,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.sampler,
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical input via a quantum circuit and decode classically."""
        batch = x.detach().cpu().numpy()
        latent = self.qnn.forward(batch)
        latent_torch = torch.as_tensor(latent, dtype=torch.float32, device=x.device)
        return self.decoder(latent_torch)


def train_autoencoder_qml(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    device: torch.device | None = None,
) -> List[float]:
    """Train the quantum autoencoder using COBYLA to optimise variational parameters."""
    device = device or torch.device("cpu")
    data_np = data.detach().cpu().numpy()
    history: List[float] = []

    for _ in range(epochs):
        def loss_fn(params: np.ndarray) -> float:
            weight_dict = {p: val for p, val in zip(model.qnn.weight_params, params)}
            latent = model.qnn.forward(data_np, weight_params=weight_dict)
            recon = model.decoder(torch.as_tensor(latent, dtype=torch.float32, device=device))
            return ((recon - data_np) ** 2).mean().item()

        init_params = np.zeros(len(model.qnn.weight_params))
        res = COBYLA(fun=loss_fn, x0=init_params, maxfun=200)
        history.append(loss_fn(res.x))
    return history


__all__ = ["HybridAutoencoder", "train_autoencoder_qml", "FraudLayerParameters"]
