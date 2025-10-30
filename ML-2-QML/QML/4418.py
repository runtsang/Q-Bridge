"""Hybrid quantum auto‑encoder that integrates real Qiskit circuits with classical layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import CircuitQNN


class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter used as a drop‑in replacement for a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # collapse spatial dims


class QuantumAutoEncoderCircuit(nn.Module):
    """Parameterised quantum circuit that maps a latent vector to a quantum latent representation."""
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.weight_params = nn.Parameter(torch.randn(latent_dim))
        self.params = ParameterVector("params", 2 * latent_dim)

        circuit = QuantumCircuit(latent_dim)
        for i in range(latent_dim):
            circuit.rx(self.params[i], i)                    # input rotation
            circuit.rx(self.params[latent_dim + i], i)        # weight rotation
        circuit.measure_all()

        self.qnn = CircuitQNN(
            circuit=circuit,
            input_params=[self.params],
            weight_params=[],
            output_shape=latent_dim,
            primitive=Sampler(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight_batch = self.weight_params.expand(inputs.shape[0], -1)
        params = torch.cat([inputs, weight_batch], dim=1)
        out = self.qnn(params)
        return torch.sigmoid(out)


class QuantumGate(nn.Module):
    """Quantum gate implemented as a parameterised circuit returning a vector of activations."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.params = ParameterVector("params", 2 * n_qubits)

        circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            circuit.rx(self.params[i], i)                    # input rotation
            circuit.rx(self.params[n_qubits + i], i)          # weight rotation
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        circuit.cx(n_qubits - 1, 0)
        circuit.measure_all()

        self.qnn = CircuitQNN(
            circuit=circuit,
            input_params=[self.params],
            weight_params=[],
            output_shape=n_qubits,
            primitive=Sampler(),
        )

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        params = torch.cat([inputs, weights], dim=1)
        out = self.qnn(params)
        return torch.sigmoid(out)


class QuantumLSTMCell(nn.Module):
    """Quantum LSTM cell where each gate is a small quantum circuit."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.forget_gate = QuantumGate(n_qubits)
        self.input_gate = QuantumGate(n_qubits)
        self.update_gate = QuantumGate(n_qubits)
        self.output_gate = QuantumGate(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.n_qubits, device=inputs.device)
        cx = torch.zeros(batch_size, self.n_qubits, device=inputs.device)
        if states is not None:
            hx, cx = states
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined[:, :self.n_qubits], combined[:, self.n_qubits:]))
            i = torch.sigmoid(self.input_gate(combined[:, :self.n_qubits], combined[:, self.n_qubits:]))
            g = torch.tanh(self.update_gate(combined[:, :self.n_qubits], combined[:, self.n_qubits:]))
            o = torch.sigmoid(self.output_gate(combined[:, :self.n_qubits], combined[:, self.n_qubits:]))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid quantum‑classical auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    conv_kernel: int = 2
    conv_threshold: float = 0.0
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quantum_ae: bool = True
    use_quantum_lstm: bool = True


class HybridAutoencoder(nn.Module):
    """Hybrid quantum‑classical auto‑encoder with real Qiskit circuits."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = ConvFilter(kernel_size=cfg.conv_kernel, threshold=cfg.conv_threshold)

        # Linear encoder to match the latent dimension
        conv_dim = cfg.input_dim * cfg.conv_kernel * cfg.conv_kernel
        self.linear_encoder = nn.Linear(conv_dim, cfg.latent_dim)

        # Quantum auto‑encoder circuit
        self.quantum_ae = QuantumAutoEncoderCircuit(cfg.latent_dim) if cfg.use_quantum_ae else None

        # Quantum LSTM cell
        self.quantum_lstm = QuantumLSTMCell(cfg.latent_dim) if cfg.use_quantum_lstm else None

        # Linear decoder
        self.linear_decoder = nn.Linear(cfg.latent_dim, conv_dim)

        # Optional classical decoder layers mirroring the classical counterpart
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, conv_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        conv_out = self.conv(x).view(x.size(0), -1)
        latent = self.linear_encoder(conv_out)

        if self.quantum_ae is not None:
            latent = self.quantum_ae(latent)

        if self.quantum_lstm is not None:
            latent_seq = latent.unsqueeze(0)  # seq_len = 1
            lstm_out, _ = self.quantum_lstm(latent_seq)
            latent = lstm_out.squeeze(0)
        else:
            latent = latent  # no LSTM

        recon = self.linear_decoder(latent)
        recon = self.decoder(recon)
        recon = recon.view(x.size())
        return recon


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "train_hybrid_autoencoder"]
