"""Hybrid regressor with classical backbone and quantum submodules.

This module implements a unified regressor that can operate in three modes:
  - classic: plain feed‑forward network.
  - attention_q: classical backbone + quantum multi‑head attention block.
  - quantum_ffn: classical backbone + quantum feed‑forward block.

The quantum blocks are implemented with torchquantum and use a state‑vector
device for simulation. They can be replaced with real quantum hardware by
changing the device configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class _QuantumAttention(tq.QuantumModule):
    """Quantum multi‑head attention implemented as a variational circuit."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        return self.measure(q_device)

class _QuantumFFN(tq.QuantumModule):
    """Quantum feed‑forward block realised by a variational circuit."""
    def __init__(self, n_qubits: int, ffn_dim: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.ffn_dim = ffn_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        return self.measure(q_device)

class UnifiedRegressorQNN(nn.Module):
    """Hybrid regressor combining classical backbone with optional quantum modules."""
    def __init__(self,
                 mode: str = "classic",
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 n_heads: int = 2,
                 n_qubits: int = 4,
                 q_device: tq.QuantumDevice | None = None):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)

        # Classical backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if mode == "classic":
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif mode == "attention_q":
            self.attention = _QuantumAttention(n_qubits)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif mode == "quantum_ffn":
            self.quantum_ffn = _QuantumFFN(n_qubits, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone and the chosen quantum block."""
        x = self.backbone(x)
        if self.mode == "classic":
            return self.output_layer(x)
        elif self.mode == "attention_q":
            batch_size = x.size(0)
            qdev = self.q_device.copy(bsz=batch_size, device=x.device)
            q_out = self.attention(x, qdev)
            return self.output_layer(q_out)
        elif self.mode == "quantum_ffn":
            batch_size = x.size(0)
            qdev = self.q_device.copy(bsz=batch_size, device=x.device)
            q_out = self.quantum_ffn(x, qdev)
            return self.output_layer(q_out)

__all__ = ["UnifiedRegressorQNN"]
