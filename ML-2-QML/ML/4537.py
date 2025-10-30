"""Hybrid classical-quantum LSTM model with QCNN feature extraction and versatile heads.

This module implements a unified architecture that can be used for sequence tagging,
regression, or binary classification.  The core is a quantum‑LSTM cell that can be
switched to a classical LSTM, and the feature extraction is performed by a
lightweight QCNN‑style network.  A hybrid head replaces the quantum circuit in
the original model, while a plain linear head is available for regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  QCNN‑style feature extractor (classical)
# --------------------------------------------------------------------------- #
class QCNNFeatureExtractor(nn.Module):
    """A lightweight, fully‑connected approximation of the QCNN used in the quantum reference."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


# --------------------------------------------------------------------------- #
#  Classical LSTM cell (fallback)
# --------------------------------------------------------------------------- #
class ClassicalLSTMCell(nn.Module):
    """Simple LSTM cell using linear gates – the classical counterpart of the quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_linear(combined))
        i = torch.sigmoid(self.input_linear(combined))
        g = torch.tanh(self.update_linear(combined))
        o = torch.sigmoid(self.output_linear(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


# --------------------------------------------------------------------------- #
#  Hybrid activation (classical)
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid with a tunable shift – mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class Hybrid(nn.Module):
    """Linear head followed by a shift‑controlled sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --------------------------------------------------------------------------- #
#  Regression head
# --------------------------------------------------------------------------- #
class RegressionHead(nn.Module):
    """Simple linear regression head."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Unified LSTM model
# --------------------------------------------------------------------------- #
class QLSTMGen224(nn.Module):
    """Hybrid LSTM model that can operate in classical or quantum mode.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Hidden size of the LSTM.
    n_qubits : int
        If >0, the LSTM cell will be quantum; otherwise classical.
    task : str
        One of ``'tagger'``, ``'regression'`` or ``'binary'``.
    feature_dim : int
        Dimensionality of the raw input that is fed into the QCNN feature extractor.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        task: str = "tagger",
        feature_dim: int = 8,
    ) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(feature_dim)
        self.lstm_cell = ClassicalLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Choose head
        if task == "tagger":
            self.head = nn.Linear(hidden_dim, 1)  # placeholder; overridden in wrapper
        elif task == "regression":
            self.head = RegressionHead(hidden_dim)
        elif task == "binary":
            self.head = Hybrid(hidden_dim, shift=0.0)
        else:
            raise ValueError(f"Unsupported task {task}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (seq_len, batch, feature_dim)
        seq_len, batch, _ = inputs.shape
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)

        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            x = self.feature_extractor(x)
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Flatten for head
        lstm_out_flat = lstm_out.view(-1, self.hidden_dim)
        out = self.head(lstm_out_flat)
        return out.view(seq_len, batch, -1)


# --------------------------------------------------------------------------- #
#  Convenience wrappers
# --------------------------------------------------------------------------- #
class LSTMTaggerGen224(QLSTMGen224):
    """Tagger that returns log‑softmax over a tagset."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, tagset_size=10, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="tagger", **kwargs)
        self.head = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        lstm_out = super().forward(sentence)
        return F.log_softmax(lstm_out, dim=-1)


class RegressionGen224(QLSTMGen224):
    """Regression model that outputs a scalar."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="regression", **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out = super().forward(inputs)
        return lstm_out.squeeze(-1)


class BinaryClassifierGen224(QLSTMGen224):
    """Binary classifier that returns a probability pair."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="binary", **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = super().forward(inputs)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "QLSTMGen224",
    "LSTMTaggerGen224",
    "RegressionGen224",
    "BinaryClassifierGen224",
    "Hybrid",
    "HybridFunction",
    "RegressionHead",
]
