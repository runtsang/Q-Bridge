import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Quantum modules
from qml_code import QuantumPhotonicCircuit, QuantumLSTM, QuantumFullyConnected, QuantumSampler

@dataclass
class FraudLayerParams:
    weight: torch.Tensor
    bias: torch.Tensor
    scale: float = 1.0
    shift: float = 0.0

def _dense_layer(params: FraudLayerParams) -> nn.Module:
    layer = nn.Linear(params.weight.shape[1], params.weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(params.weight)
        layer.bias.copy_(params.bias)
    return layer

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that blends a classical dense backbone with
    quantum photonic, quantum‑enhanced LSTM, fully‑connected and sampler blocks.
    The model can operate in three regimes:
    * classical‑only (use_quantum=False)
    * quantum‑only (classical backbone replaced by a linear layer)
    * hybrid (both parts contribute)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 4,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum

        # Classical dense backbone
        self.classical_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.classical_out = nn.Linear(hidden_dim, 1)

        if use_quantum:
            self.quantum_photonic = QuantumPhotonicCircuit(n_modes=2)
            self.quantum_lstm = QuantumLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, n_qubits=n_qubits)
            self.quantum_fully = QuantumFullyConnected(n_qubits=n_qubits)
            self.quantum_sampler = QuantumSampler(n_qubits=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        h = self.classical_layers(x)
        h = h + self.residual(h)
        class_out = self.classical_out(h)  # shape (batch, 1)

        if not self.use_quantum:
            return torch.sigmoid(class_out)

        # Quantum path
        # Photonic circuit
        photonic_np = self.quantum_photonic.run(h.detach().cpu().numpy())
        photonic_out = torch.tensor(photonic_np, dtype=torch.float32, device=h.device)

        # Quantum LSTM
        lstm_out_seq, _ = self.quantum_lstm(h.unsqueeze(0))
        lstm_out = lstm_out_seq.mean(dim=0).mean(dim=-1, keepdim=True)  # shape (batch, 1)

        # Quantum fully‑connected
        fc_np = self.quantum_fully.run(h.detach().cpu().numpy())
        fc_out = torch.tensor(fc_np, dtype=torch.float32, device=h.device)

        # Quantum sampler
        sampler_np = self.quantum_sampler.run(h.detach().cpu().numpy())
        sampler_out = torch.tensor(sampler_np, dtype=torch.float32, device=h.device)

        # Concatenate all signals
        combined = torch.cat([class_out, photonic_out, lstm_out, fc_out, sampler_out], dim=-1)
        return torch.sigmoid(combined.sum(dim=-1, keepdim=True))

__all__ = ["FraudDetectionHybrid"]
