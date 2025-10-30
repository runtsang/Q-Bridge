from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class QLSTMGen(nn.Module):
    """Hybrid LSTM that augments a classical LSTM cell with a learnable phase‑shift matrix
    and a lightweight GRU gate. The phase matrix is applied to the hidden state before
    the LSTM update, allowing the model to capture quantum‑inspired correlations while
    remaining fully differentiable on any PyTorch backend.

    The class is a drop‑in replacement for the original pure‑classical QLSTM.  It supports
    optional gradient checkpointing for memory‑efficient training and a configurable
    quantum‑to‑classical feature extractor that can be toggled during inference.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_checkpoint: bool = False,
        feature_extraction: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_checkpoint = use_checkpoint
        self.feature_extraction = feature_extraction

        # Classical LSTM cell that will be wrapped with a phase‑shift layer
        self.lstm_cell = nn.LSTMCell(input_dim + hidden_dim, hidden_dim)

        # Learnable phase‑shift matrix (only active when n_qubits > 0)
        if n_qubits > 0:
            self.phase = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        else:
            self.phase = None

        # Classical GRU gate to fuse the hidden state with the phase‑shifted output
        self.gru_gate = nn.GRUCell(hidden_dim, hidden_dim)

        # Optional feature extractor mapping qubit outputs back to hidden_dim
        if feature_extraction:
            self.feature_extractor = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.feature_extractor = None

    def _apply_phase(self, hx: torch.Tensor) -> torch.Tensor:
        """Apply the learned phase‑shift matrix to the hidden state."""
        if self.phase is None:
            return hx
        return torch.matmul(hx, self.phase)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Optionally use checkpointing to save memory
            if self.use_checkpoint:
                hx, cx = checkpoint(self.lstm_cell, combined, (hx, cx))
            else:
                hx, cx = self.lstm_cell(combined, (hx, cx))

            # Phase‑shift the hidden state
            hx = self._apply_phase(hx)

            # Optional feature extraction
            if self.feature_extractor is not None:
                hx = self.feature_extractor(hx)

            # Fuse with a lightweight GRU gate
            hx = self.gru_gate(hx, hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical LSTM, the hybrid QLSTMGen,
    or a pure quantum variant (when used in the quantum module).  The interface is
    preserved so that the tagger can be used as a drop‑in replacement in existing pipelines.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_checkpoint: bool = False,
        feature_extraction: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(
                embedding_dim,
                hidden_dim,
                n_qubits,
                use_checkpoint=use_checkpoint,
                feature_extraction=feature_extraction,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTagger"]
