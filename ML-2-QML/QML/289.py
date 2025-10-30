from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class QuantumAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    latent_dim: int
    num_features: int
    reps: int = 3


class HybridAutoencoder(nn.Module):
    """Quantum autoencoder based on a variational RealAmplitudes circuit."""
    def __init__(self, config: QuantumAutoencoderConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.num_features = config.num_features
        self.reps = config.reps

        num_qubits = self.latent_dim + self.num_features

        # Input parameters: one per feature
        self.input_params = [Parameter(f"x{i}") for i in range(self.num_features)]
        # Weight parameters: remaining parameters of the ansatz
        ansatz = RealAmplitudes(num_qubits, reps=self.reps)
        ansatz_params = list(ansatz.parameters)
        self.weight_params = [Parameter(f"w{j}") for j in range(len(ansatz_params))]

        # Build circuit
        circuit = QuantumCircuit(num_qubits)
        for i, param in enumerate(self.input_params):
            circuit.rx(param, i)

        # Map ansatz parameters to weight_params
        param_map = {ansatz_params[j]: self.weight_params[j] for j in range(len(ansatz_params))}
        circuit.compose(ansatz, inplace=True, front=False, param_map=param_map)

        # Sampler QNN
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=self.num_features,
            sampler=Sampler(),
        )

        # Initialize weight values
        self.weight_values = nn.Parameter(torch.randn(len(self.weight_params)))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode classical data into quantum outputs."""
        inputs_np = inputs.detach().cpu().numpy()
        weight_np = self.weight_values.detach().cpu().numpy()
        outputs = self.qnn(inputs_np, weights=weight_np)
        return torch.tensor(outputs, dtype=torch.float32, device=inputs.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """For a quantum autoencoder, decoding is identical to encoding."""
        return self.encode(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)


def QuantumAutoencoder(
    latent_dim: int,
    num_features: int,
    *,
    reps: int = 3,
) -> HybridAutoencoder:
    config = QuantumAutoencoderConfig(
        latent_dim=latent_dim,
        num_features=num_features,
        reps=reps,
    )
    return HybridAutoencoder(config)


def train_qml_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> list[float]:
    """Train the quantum autoencoder using a classical optimizer."""
    if device!= "cpu":
        model.to(device)
    optimizer = torch.optim.Adam([model.weight_values], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model.encode(data)
        loss = loss_fn(outputs, data)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


__all__ = ["HybridAutoencoder", "QuantumAutoencoder", "train_qml_autoencoder"]
