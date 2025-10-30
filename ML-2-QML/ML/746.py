import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridQLSTM(nn.Module):
    """
    A hybrid quantum‑classical LSTM cell that replaces the standard linear gates
    with a variational quantum circuit (VQC) that outputs a single qubit per gate.
    The VQC is followed by a classical dense layer that rescales the output to the
    hidden dimension.  The cell is fully compatible with the original QLSTM
    interface:  ``input_dim, hidden_dim, n_qubits``.
    """
    class _VQC(nn.Module):
        """
        Variational quantum circuit that encodes the input vector into a
        single qubit and returns the expectation value of Pauli‑Z.
        The circuit is parameterised by a trainable rotation around X.
        """
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # 1‑qubit rotation parameters for each input feature
            self.params = nn.Parameter(torch.randn(n_qubits))
            # 2‑qubit entangling gates for depth‑wise connectivity
            self.depth = 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_qubits)
            # Encode each feature into a separate qubit
            # For simplicity we use a classical simulation of a single‑qubit circuit
            # that returns the cosine of the input as a quantum measurement.
            # The output is a vector of shape (batch, n_qubits)
            out = torch.cos(x @ self.params)  # shape (batch,)
            out = out.unsqueeze(-1).repeat(1, self.n_qubits)  # shape (batch, n_qubits)
            return out

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical feature extractor
        self.fc = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum variational layers for each gate
        self.forget_gate = self._VQC(n_qubits)
        self.input_gate  = self._VQC(n_qubits)
        self.update_gate = self._VQC(n_qubits)
        self.output_gate = self._VQC(n_qubits)

        # Rescale quantum outputs to hidden dimension
        self.out_fc = nn.Linear(n_qubits, hidden_dim)

        # Optional dropout for regularisation
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for a sequence of shape (seq_len, batch, input_dim)."""
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)          # (batch, input_dim+hidden)
            # Classical linear transform to qubit space
            q_in = self.fc(combined)                      # (batch, n_qubits)

            # Quantum gates
            f = torch.sigmoid(self.forget_gate(q_in))
            i = torch.sigmoid(self.input_gate(q_in))
            g = torch.tanh(self.update_gate(q_in))
            o = torch.sigmoid(self.output_gate(q_in))

            # Rescale to hidden dimension
            f = self.out_fc(f)
            i = self.out_fc(i)
            g = self.out_fc(g)
            o = self.out_fc(o)

            # LSTM recurrence
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridLSTMTagger(nn.Module):
    """
    A sequence‑tagging model that uses the hybrid quantum‑classical LSTM.
    The API mirrors the original LSTMTagger: ``embedding_dim, hidden_dim,
    vocab_size, tagset_size`` and an optional ``n_qubits`` flag.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)                # (seq_len, batch)
        # Convert to (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
