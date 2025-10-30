"""Quantum‑only version of the hybrid vision‑to‑sequence model.

Uses Pennylane to implement a quantum quanvolution filter and a quantum‑enhanced LSTM.
The model processes 28×28 grayscale images by extracting 2×2 patches, encoding each patch
into a 4‑qubit circuit, and feeding the resulting sequence into a quantum LSTM that
propagates information across the patch sequence. The final hidden state is mapped
to class logits via a classical linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple

class QLayer(nn.Module):
    """Small quantum circuit used as a gate in the quantum LSTM."""
    def __init__(self, n_wires: int, device: str = "default.qubit", shots: int = 1024) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device(device, wires=n_wires, shots=shots)
        self.params = nn.Parameter(torch.randn(n_wires))
        self.circuit = qml.qnode(self._circuit, device=self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[float]:
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
        for i in range(self.n_wires):
            qml.RX(params[i], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x, self.params).unsqueeze(-1)

class QuantumQLSTM(nn.Module):
    """Quantum‑enhanced LSTM that replaces each gate with a QLayer."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 device: str = "default.qubit", shots: int = 1024) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits, device, shots)
        self.input = QLayer(n_qubits, device, shots)
        self.update = QLayer(n_qubits, device, shots)
        self.output = QLayer(n_qubits, device, shots)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QuantumQuanvolutionFilter(nn.Module):
    """Quantum filter that processes 2×2 image patches with a 4‑qubit circuit."""
    def __init__(self, n_wires: int = 4, patch_size: int = 2,
                 device: str = "default.qubit", shots: int = 1024) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        self.dev = qml.device(device, wires=n_wires, shots=shots)
        self.params = nn.Parameter(torch.randn(n_wires))
        self.circuit = qml.qnode(self._circuit, device=self.dev, interface="torch")

    def _circuit(self, patch: torch.Tensor, params: torch.Tensor) -> list[float]:
        for i in range(self.n_wires):
            qml.RY(patch[i], wires=i)
        for i in range(self.n_wires):
            qml.RX(params[i], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, :, r:r + self.patch_size, c:c + self.patch_size]
                patch = patch.view(batch_size, -1)  # (B, 4)
                feat = self.circuit(patch, self.params)
                patches.append(feat)
        return torch.cat(patches, dim=1)

class QuanvolutionQLSTMHybrid(nn.Module):
    """Quantum vision‑to‑sequence model that fuses a quantum quanvolution filter
    with a quantum‑enhanced LSTM and a classical linear classifier."""
    def __init__(self, n_qubits: int = 4, hidden_dim: int = 128,
                 num_classes: int = 10,
                 device: str = "default.qubit", shots: int = 1024) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(n_qubits, device=device, shots=shots)
        self.lstm = QuantumQLSTM(input_dim=4, hidden_dim=hidden_dim,
                                 n_qubits=n_qubits, device=device, shots=shots)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        patches = self.qfilter(x)  # (B, 4*14*14)
        seq_len = 14 * 14
        patch_dim = 4
        patches_seq = patches.view(batch_size, seq_len, patch_dim).transpose(0, 1)  # (seq_len, B, 4)
        lstm_out, _ = self.lstm(patches_seq)
        final_hidden = lstm_out[-1]  # (B, hidden_dim)
        logits = self.classifier(final_hidden)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QLayer", "QuantumQLSTM", "QuantumQuanvolutionFilter", "QuanvolutionQLSTMHybrid"]
