"""QuantumHybridNAT: Classical backbone with optional quantum augmentation.

This module implements a hybrid architecture that combines a CNN‑FC encoder,
an optional RBF kernel head, and a sequence module that can be an LSTM,
a quantum‑LSTM (QLSTM) or a transformer.  The design is inspired by
the Quantum‑NAT, QuantumKernelMethod, QLSTM and QTransformerTorch
reference seeds.

Author: OpenAI GPT‑oss‑20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

# --------------------------------------------------------------------------- #
# 1. Classical CNN‑FC encoder
# --------------------------------------------------------------------------- #
class _CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 2. Classical RBF kernel
# --------------------------------------------------------------------------- #
class _Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# 3. Classical QLSTM (from reference 3)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# 4. Simple transformer (classical)
# --------------------------------------------------------------------------- #
class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim: int = 1, num_heads: int = 1, ffn_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

# --------------------------------------------------------------------------- #
# 5. Hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridNAT(nn.Module):
    """
    A hybrid architecture that combines a classical CNN‑FC encoder,
    a kernel head, and a sequence module that can be an LSTM, a quantum‑LSTM
    (QLSTM) or a transformer.  The model is fully torch‑compatible and
    can be used for classification tasks.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Encoder
        self.encoder = _CNNEncoder(
            in_channels=config.get("in_channels", 1),
            out_features=config.get("out_features", 4)
        )
        # Kernel
        self.kernel = _Kernel(gamma=config.get("gamma", 1.0))
        # Prototypes
        n_prototypes = config.get("n_prototypes", 10)
        feature_dim = config.get("out_features", 4)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, feature_dim))
        # Sequence module
        seq_type = config.get("seq_type", "lstm")  # options: lstm, qlstm, transformer
        seq_params = config.get("seq_params", {})
        if seq_type == "lstm":
            self.seq_module = nn.LSTM(
                input_size=1,
                hidden_size=seq_params.get("hidden_dim", 32),
                num_layers=1,
                batch_first=True
            )
            self.hidden_dim = seq_params.get("hidden_dim", 32)
        elif seq_type == "qlstm":
            self.seq_module = QLSTM(
                input_dim=1,
                hidden_dim=seq_params.get("hidden_dim", 32),
                n_qubits=seq_params.get("n_qubits", 4)
            )
            self.hidden_dim = seq_params.get("hidden_dim", 32)
        elif seq_type == "transformer":
            self.seq_module = SimpleTransformer(
                embed_dim=1,
                num_heads=seq_params.get("num_heads", 1),
                ffn_dim=seq_params.get("ffn_dim", 32),
                dropout=seq_params.get("dropout", 0.1)
            )
            self.hidden_dim = 1
        else:
            raise ValueError(f"Unsupported seq_type {seq_type}")
        # Final classifier
        self.classifier = nn.Linear(self.hidden_dim, config.get("num_classes", 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor of shape [batch, channels, height, width]
        Output:
            logits: Tensor of shape [batch, num_classes]
        """
        # Encode
        features = self.encoder(x)  # [batch, feature_dim]
        # Compute kernel similarities with prototypes
        sims = torch.stack([self.kernel(features, proto) for proto in self.prototypes], dim=1)  # [batch, n_prototypes]
        # Prepare sequence input: [batch, seq_len, input_dim=1]
        seq_input = sims.unsqueeze(-1)
        # Sequence module
        if isinstance(self.seq_module, nn.LSTM):
            out, _ = self.seq_module(seq_input)
            hidden = out[:, -1, :]  # last hidden state
        elif isinstance(self.seq_module, QLSTM):
            out, _ = self.seq_module(seq_input.permute(1, 0, 2))
            hidden = out[-1, :, :]  # last hidden state
        else:  # transformer
            hidden = self.seq_module(seq_input)[:, -1, :]  # take last token
        # Classify
        logits = self.classifier(hidden)
        return logits
