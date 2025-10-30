"""QML module for HybridBinaryClassifier.

This module replaces the final linear layer with a variational quantum circuit
or an EstimatorQNN, while keeping the classical backbone unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Auto‑encoder definitions (identical to the classical module)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory returning a configured auto‑encoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# --------------------------------------------------------------------------- #
# Quantum circuit wrapper (from reference 1)
# --------------------------------------------------------------------------- #
from qiskit import Aer, QuantumCircuit, assemble, transpile

class QuantumCircuitWrapper:
    """Parameterized two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = QuantumCircuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# Hybrid function connecting PyTorch and the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

# --------------------------------------------------------------------------- #
# Hybrid layer (from reference 1)
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# Estimator‑QNN wrapper (from reference 4)
# --------------------------------------------------------------------------- #
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

class EstimatorQNNQuantum(nn.Module):
    """Wrapper for Qiskit’s EstimatorQNN performing regression."""
    def __init__(self) -> None:
        super().__init__()
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)

        observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=self.estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        inp_np = inputs.detach().cpu().numpy()
        result = self.estimator_qnn.predict(inp_np)
        return torch.tensor(result, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Main hybrid classifier (quantum head)
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    Convolutional backbone + optional auto‑encoder + quantum hybrid head.
    Mirrors the classical version but replaces the final linear layer
    with a parameterised circuit or an EstimatorQNN.
    """
    def __init__(
        self,
        use_autoencoder: bool = False,
        use_estimator: bool = False,
        autoencoder_config: AutoencoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.use_autoencoder = use_autoencoder
        self.use_estimator = use_estimator

        # Convolutional backbone (identical to the classical one)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Optional auto‑encoder
        if self.use_autoencoder:
            cfg = (
                autoencoder_config
                or AutoencoderConfig(input_dim=84, latent_dim=32, hidden_dims=(64, 32), dropout=0.1)
            )
            self.autoencoder = Autoencoder(cfg.input_dim)
        else:
            self.autoencoder = None

        # Quantum or Estimator head
        backend = Aer.get_backend("aer_simulator")
        if self.use_estimator:
            self.head = EstimatorQNNQuantum()
        else:
            self.head = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.autoencoder is not None:
            x = self.autoencoder(x)

        logits = self.head(x)
        # Return binary probabilities
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = [
    "HybridBinaryClassifier",
    "QuantumCircuitWrapper",
    "HybridFunction",
    "Hybrid",
    "EstimatorQNNQuantum",
]
