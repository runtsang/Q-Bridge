"""Hybrid quantum/classical LSTM with optional quantum feature extraction and attention.

This module defines :class:`HybridQLSTM`, mirroring the classical counterpart but
leveraging `torchquantum` for quantum gates and `qiskit` for QCNN.  The design keeps
the same API as the original `QLSTM` while providing a richer set of
quantum‑enabled building blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

import torchquantum as tq
import torchquantum.functional as tqf

# Optional imports
try:
    from QCNN import QCNN  # quantum QCNN function
except Exception:
    QCNN = None

try:
    from Quanvolution import QuanvolutionFilter  # quantum quanvolution
except Exception:
    QuanvolutionFilter = None

try:
    from QTransformerTorch import TransformerBlockQuantum
except Exception:
    TransformerBlockQuantum = None

class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM model that supports:
        * Quantum LSTM gates via torchquantum
        * Optional quantum Transformer block
        * Optional quantum QCNN or Quanvolution feature extractor
    """

    class QLayer(tq.QuantumModule):
        """Reusable quantum layer for linear‑like transformations."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_transformer: bool = False,
        transformer_params: Optional[Dict] = None,
        use_qcnn: bool = False,
        use_quanvolution: bool = False,
        num_classes: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.device = device

        # Feature extractor
        self.feature_extractor = None
        if use_qcnn and QCNN is not None:
            self.feature_extractor = QCNN()
        elif use_quanvolution and QuanvolutionFilter is not None:
            self.feature_extractor = QuanvolutionFilter()

        # Quantum LSTM gates
        self.forget = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output_gate = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Optional Transformer
        self.transformer = None
        if use_transformer and TransformerBlockQuantum is not None:
            tp = transformer_params or {}
            embed_dim = tp.get("embed_dim", hidden_dim)
            num_heads = tp.get("num_heads", 8)
            ffn_dim = tp.get("ffn_dim", hidden_dim * 4)
            n_qubits_transformer = tp.get("n_qubits_transformer", n_qubits)
            self.transformer = TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits, 1, None, 0.1
            )

        # Classifier head
        if num_classes > 2:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)

    def _init_states(self, inputs: torch.Tensor, states: Optional[tuple] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            log‑probabilities of shape (batch, num_classes)
        """
        # Feature extraction
        if self.feature_extractor is not None:
            batch, seq_len, _ = x.size()
            x_flat = x.contiguous().view(batch * seq_len, -1)
            feats = self.feature_extractor(x_flat)
            feats = feats.view(batch, seq_len, -1)
            x = feats

        # Recurrent processing
        hx, cx = self._init_states(x)
        outputs = []
        qdev = tq.QuantumDevice(self.n_qubits, bsz=hx.size(0), device=hx.device)
        for t in range(x.size(1)):
            xt = x[:, t, :]
            combined = torch.cat([xt, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined), qdev))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined), qdev))
            g = torch.tanh(self.update(self.linear_update(combined), qdev))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined), qdev))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        lstm_out = torch.cat(outputs, dim=1)

        # Optional Transformer refinement
        if self.transformer is not None:
            lstm_out = self.transformer(lstm_out)

        # Classification
        logits = self.classifier(lstm_out.mean(dim=1))
        return F.log_softmax(logits, dim=-1)
