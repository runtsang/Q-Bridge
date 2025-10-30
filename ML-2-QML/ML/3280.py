"""Unified QCNN‑QLSTM hybrid for classical‑plus‑quantum experiments.

The module defines a hybrid architecture that combines a dense‑layer
convolution‑style network (ClassicalQCNN) with either a classical or a
quantum LSTM cell.  The classical branch uses a straight‑forward
PyTorch implementation that mirrors the original QCNN seed.  The quantum
branch is provided in the separate QML module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Classical convolution‑style network (mirrors QCNN.py)
class ClassicalQCNN(nn.Module):
    """Dense‑layer CNN‑like model that emulates the QCNN feature‑map,
    convolution and pooling steps."""
    def __init__(self, in_features: int = 8, hidden: int = 16,
                 pool_size: int = 12, conv_size: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden, pool_size), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(pool_size, conv_size), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(conv_size, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Classical LSTM cell (drop‑in replacement)
class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# Unified hybrid model
class UnifiedQCNNQLSTM(nn.Module):
    """End‑to‑end model that optionally uses either classical or quantum QCNN
    followed by either classical or quantum LSTM for sequence tagging."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        vocab_size: int = 0,
        tagset_size: int = 0,
        use_quantum_qcnn: bool = False,
        use_quantum_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.use_quantum_qcnn = use_quantum_qcnn
        self.use_quantum_lstm = use_quantum_lstm

        # QCNN branch
        if use_quantum_qcnn:
            try:
                from.qml_code import QuantumQCNN  # type: ignore
                self.qcnn = QuantumQCNN()
            except Exception:
                raise RuntimeError("QuantumQCNN not available in ml module")
        else:
            self.qcnn = ClassicalQCNN(in_features=input_dim)

        # LSTM branch
        if use_quantum_lstm:
            try:
                from.qml_code import QuantumQLSTM  # type: ignore
                self.lstm = QuantumQLSTM(input_dim=1, hidden_dim=hidden_dim, n_qubits=4)
            except Exception:
                raise RuntimeError("QuantumQLSTM not available in ml module")
        else:
            self.lstm = ClassicalQLSTM(input_dim=1, hidden_dim=hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence tagging.

        Args:
            sentence: Tensor of shape (seq_len, batch, input_dim) where each
            time‑step is a feature vector of length `input_dim`.

        Returns:
            Log‑softmax over tagset for each time‑step.
        """
        seq_len, batch, _ = sentence.size()
        flattened = sentence.view(seq_len * batch, -1)
        qcnn_out = self.qcnn(flattened)  # shape (seq_len*batch, 1)
        qcnn_out = qcnn_out.view(seq_len, batch, -1)
        lstm_out, _ = self.lstm(qcnn_out)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["UnifiedQCNNQLSTM", "ClassicalQCNN", "ClassicalQLSTM"]
