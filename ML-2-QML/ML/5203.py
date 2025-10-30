"""Hybrid LSTM implementation with optional classical and quantum feature extraction and attention.

This module defines :class:`HybridQLSTM`, a versatile sequence model that can be configured to use
classical LSTM gates, Transformer blocks, and optional QCNN or Quanvolution feature extractors.
The design follows the API of the original `QLSTM` seed while adding modularity and scalability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Optional feature extractor imports
try:
    from QCNN import QCNNModel
except Exception:
    QCNNModel = None

try:
    from Quanvolution import QuanvolutionFilter
except Exception:
    QuanvolutionFilter = None

# Optional transformer imports
try:
    from QTransformerTorch import TransformerBlockClassical
except Exception:
    TransformerBlockClassical = None

class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM model that supports:
        * Classical LSTM gates or quantum gates (via torchquantum)
        * Optional Transformer block (classical)
        * Optional QCNN or Quanvolution feature extractor
    The class is intentionally lightweight and can be used as a drop‑in replacement
    for the original `QLSTM` implementation.
    """

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
        if use_qcnn and QCNNModel is not None:
            self.feature_extractor = QCNNModel()
        elif use_quanvolution and QuanvolutionFilter is not None:
            self.feature_extractor = QuanvolutionFilter()

        # Main recurrent layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Optional Transformer
        self.transformer = None
        if use_transformer and TransformerBlockClassical is not None:
            tp = transformer_params or {}
            embed_dim = tp.get("embed_dim", hidden_dim)
            num_heads = tp.get("num_heads", 8)
            ffn_dim = tp.get("ffn_dim", hidden_dim * 4)
            self.transformer = TransformerBlockClassical(embed_dim, num_heads, ffn_dim)

        # Classifier head
        if num_classes > 2:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)

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
        lstm_out, _ = self.lstm(x)

        # Optional Transformer refinement
        if self.transformer is not None:
            lstm_out = self.transformer(lstm_out)

        # Classification
        logits = self.classifier(lstm_out.mean(dim=1))
        return F.log_softmax(logits, dim=-1)
