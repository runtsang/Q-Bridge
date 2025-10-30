"""Hybrid LSTM with optional classical QCNN feature extractor.

This module defines a single :class:`HybridQLSTM` class that can operate in
two regimes:

* Classical: Uses a standard PyTorch LSTM together with a fully‑connected
  QCNN‑style feature extractor (``ClassicalQCNNModel``).
* Quantum: (Implemented in the separate QML module) replaces the gates of
  the LSTM by small quantum circuits and the QCNN feature extractor by a
  parameterised quantum circuit.

The interface mirrors the original ``QLSTM`` and ``LSTMTagger`` classes
so that existing training scripts continue to work unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical LSTM cell – gates are realised by linear layers.
class ClassicalQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Classical QCNN‑style feature extractor
class ClassicalQCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Unified hybrid model
class HybridQLSTM(nn.Module):
    """Drop‑in replacement that can switch between classical and quantum back‑ends.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embeddings.
    hidden_dim : int
        Hidden size of the LSTM layer.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of distinct tags.
    use_qcircuit : bool, optional
        If ``True`` the LSTM gates are realised by quantum circuits
        (implemented in the QML module).  Otherwise a standard
        :class:`torch.nn.LSTM` is used.
    use_qconv : bool, optional
        If ``True`` the input embeddings are passed through a
        QCNN‑style feature extractor before the LSTM.
    n_qubits : int, optional
        Number of qubits used by the quantum gates (ignored in the
        classical branch).
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_qcircuit: bool = False,
        use_qconv: bool = False,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_qconv = use_qconv
        self.use_qcircuit = use_qcircuit
        if use_qconv:
            self.qconv = ClassicalQCNNModel()
        else:
            self.qconv = None
        if use_qcircuit:
            # Quantum variant is defined in the QML module.
            raise RuntimeError(
                "Quantum variant is only available in the QML module; "
                "import the QML version for quantum behaviour."
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        if self.use_qconv:
            embeds = self.qconv(embeds)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM"]
