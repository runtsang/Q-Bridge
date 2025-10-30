"""Classical hybrid LSTM module with optional QCNN feature extractor and sampler.

This module builds upon the classical LSTM implementation, extends it with a
QCNN-style fully‑connected feature extractor, and includes a lightweight
sampler network.  The interface is intentionally identical to the quantum
counterpart so that the same class name can be imported from either module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  QCNN-inspired feature extractor
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating a quantum convolutional network."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim - 4), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim - 4, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 4), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
#  Sampler network
# --------------------------------------------------------------------------- #
class SamplerModule(nn.Module):
    """Simple two‑layer sampler that produces a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Hybrid LSTM that can operate in classical or quantum mode and
    optionally use a QCNN feature extractor or a sampler.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input token.
    hidden_dim : int
        Hidden state dimensionality.
    n_qubits : int, default 0
        Number of quantum gates to use.  If zero the module falls back to a
        classical :class:`torch.nn.LSTM`.
    task : str, default 'tagging'
        Either ``'tagging'`` or ``'regression'``.
    tagset_size : int, default 10
        Number of tags for the tagging task.
    use_qcnet : bool, default False
        Whether to apply the QCNN feature extractor before the LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        task: str = "tagging",
        tagset_size: int = 10,
        use_qcnet: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.task = task
        self.use_qcnet = use_qcnet

        # Select the underlying recurrent layer
        if n_qubits == 0:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            # Placeholder; the quantum implementation will replace this
            self.lstm = None  # type: ignore[assignment]

        # Output head
        out_dim = 1 if task == "regression" else tagset_size
        self.head = nn.Linear(hidden_dim, out_dim)

        # Optional QCNN feature extractor
        self.qcnet: Optional[QCNNModel] = QCNNModel(input_dim) if use_qcnet else None

        # Sampler
        self.sampler = SamplerModule()

    # ----------------------------------------------------------------------- #
    #  Forward pass
    # ----------------------------------------------------------------------- #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for both tagging and regression.

        * Tagging: ``inputs`` shape ``(seq_len, batch, input_dim)``.
        * Regression: ``inputs`` shape ``(batch, feature_dim)``.
        """
        if self.task == "tagging":
            # Classical or quantum LSTM
            lstm_out, _ = self.lstm(inputs)
            logits = self.head(lstm_out)
            return F.log_softmax(logits, dim=-1)

        if self.task == "regression":
            # Optional QCNN preprocessing
            features = self.qcnet(inputs) if self.qcnet else inputs
            # Treat the features as a sequence of length 1
            features = features.unsqueeze(1)
            lstm_out, _ = self.lstm(features)
            return self.head(lstm_out.squeeze(1))

        raise ValueError(f"Unsupported task: {self.task}")

    # ----------------------------------------------------------------------- #
    #  Sampler interface
    # ----------------------------------------------------------------------- #
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution from the sampler."""
        return self.sampler(inputs)


__all__ = ["HybridQLSTM", "QCNNModel", "SamplerModule"]
