import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Sequence
import qiskit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudDetectionLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

class HybridFusionNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 fraud_params: Sequence[FraudLayerParameters] | None = None,
                 output_dim: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Quantum ansatz
        self.ansatz = RealAmplitudes(latent_dim, reps=5)
        self.circuit = qiskit.QuantumCircuit(latent_dim)
        self.circuit.compose(self.ansatz, range(latent_dim), inplace=True)
        # Observables: Z on each qubit
        self.observables = []
        for i in range(latent_dim):
            pauli_str = "I"*i + "Z" + "I"*(latent_dim-i-1)
            self.observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
        # Estimator
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[],
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )
        # Fraud layers
        self.fraud_layers = nn.ModuleList()
        if fraud_params:
            for p in fraud_params:
                self.fraud_layers.append(FraudDetectionLayer(p, clip=True))
        # Output layer
        self.output_layer = nn.Linear(latent_dim, output_dim)

    def run(self, params: np.ndarray) -> np.ndarray:
        """Compute the quantum latent vector for given parameters."""
        if params.ndim == 1:
            params = params.reshape(1, -1)
        q_latent = self.qnn.run(parameters=params)  # shape (batch_size, latent_dim)
        return q_latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass combining quantum and classical processing."""
        batch = x.detach().cpu().numpy()
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        q_latent = self.qnn.run(parameters=batch)  # shape (batch_size, latent_dim)
        latent = torch.as_tensor(q_latent, dtype=torch.float32, device=x.device)
        for layer in self.fraud_layers:
            latent = layer(latent)
        return self.output_layer(latent)

__all__ = ["HybridFusionNet", "FraudLayerParameters", "FraudDetectionLayer"]
