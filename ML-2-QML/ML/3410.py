"""Hybrid autoencoder combining classical convolution, quantum latent
layer, and classical decoder.

This module defines a PyTorch model that merges a convolutional feature
extractor, a quantum variational encoder (implemented with Qiskit), and a
classical linear decoder.  The design draws on the Autoencoder.py and
Conv.py seeds, extending them with a hybrid quantum–classical
architecture.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import ParameterVector

class QuantumEncoder(nn.Module):
    """PyTorch wrapper around a Qiskit `SamplerQNN` that implements a
    variational quantum encoder.
    """
    def __init__(self, num_qubits: int = 3, reps: int = 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.input_params = ParameterVector("x", num_qubits)
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        weight_params = [p for p in self.circuit.parameters if p not in self.input_params]
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=list(self.input_params),
            weight_params=weight_params,
            interpret=lambda x: x,
            output_shape=num_qubits,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qr.rx(self.input_params[i], i)
        qr.append(RealAmplitudes(self.num_qubits, reps=self.reps), range(self.num_qubits))
        return qr

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, num_qubits)
        batch = inputs.shape[0]
        outputs = torch.zeros((batch, self.num_qubits), device=inputs.device)
        for i in range(batch):
            np_input = inputs[i].detach().cpu().numpy()
            out = self.qnn.forward(np_input)
            outputs[i] = torch.tensor(out, dtype=inputs.dtype, device=inputs.device)
        return outputs


class ConvBlock(nn.Module):
    """Simple 2‑D convolution block inspired by Conv.py."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with convolutional encoder, quantum latent layer,
    and classical decoder.
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 latent_dim: int = 32,
                 conv_channels: tuple[int,...] = (8, 16),
                 num_qubits: int = 3,
                 num_trash: int = 2):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_ch = input_shape[0]
        for out_ch in conv_channels:
            enc_layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)
        dummy = torch.zeros(1, *input_shape)
        enc_out = self.encoder_conv(dummy)
        self.flatten_dim = enc_out.numel()

        # Map flattened features to latent dimension before quantum encoder
        self.fc_to_latent = nn.Linear(self.flatten_dim, num_qubits)

        # Quantum encoder
        self.quantum = QuantumEncoder(num_qubits=num_qubits)

        # Decoder
        self.decoder_fc = nn.Linear(num_qubits, self.flatten_dim)
        self.decoder_reshape = nn.Unflatten(1, input_shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        latent_pre = self.fc_to_latent(h_flat)
        latent_q = self.quantum(latent_pre)
        return latent_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        return self.decoder_reshape(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def train_hybrid_autoencoder(model: HybridAutoencoder,
                             data: torch.Tensor,
                             *,
                             epochs: int = 50,
                             batch_size: int = 32,
                             lr: float = 1e-3,
                             optimizer_class= torch.optim.Adam,
                             device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
