from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# Import the classical QCNN feature extractor (same interface as the QCNN seed)
from.QCNN import QCNNModel

class QLSTM(nn.Module):
    """Classical LSTM that can optionally augment its gates with a QCNN feature extractor."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Linear projections for the four gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional QCNN feature extractor
        if self.n_qubits > 0:
            # Map the concatenated input/hidden to the 8‑dimensional input expected by QCNNModel
            self.pre_qcnn = nn.Linear(input_dim + hidden_dim, 8)
            self.qcnn = QCNNModel()

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            # If a QCNN is present, use it to re‑weight the gates
            if self.n_qubits > 0:
                # The QCNN returns a single scalar per sample; we broadcast it to all gates
                qval = self.qcnn(self.pre_qcnn(combined))
                f = f * qval
                i = i * qval
                g = g * qval
                o = o * qval

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use a quantum‑augmented LSTM or a vanilla LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits

        if self.n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
