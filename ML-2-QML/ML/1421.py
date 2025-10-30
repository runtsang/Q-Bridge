import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Hybrid classical‑quantum LSTM cell.
    Each gate is computed by a classical linear layer and a
    toy quantum layer that simulates a variational circuit.
    The two representations are concatenated and fed through a
    combiner to produce the final gate value.
    """
    class _QuantumGate(nn.Module):
        """
        Parameter‑efficient quantum gate.
        Simulates a single‑qubit expectation value of Z after
        a parameterised RX rotation.  The rotation angle is a
        linear transformation of the input features.
        """
        def __init__(self, n_qubits: int):
            super().__init__()
            # One rotation angle per qubit
            self.params = nn.Parameter(torch.randn(n_qubits))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor of shape (batch, n_qubits)
            Returns:
                Tensor of shape (batch, n_qubits) containing
                the simulated expectation values.
            """
            # Linear mapping to rotation angles
            theta = torch.matmul(x, self.params.unsqueeze(1)).squeeze(-1)
            # Simulate expectation of Z after RX(theta)
            # 〈Z〉 = cos(2θ)
            return torch.cos(2 * theta).unsqueeze(-1)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = device or torch.device("cpu")

        # Classical linear layers for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate modules
        self.forget_q = self._QuantumGate(n_qubits)
        self.input_q = self._QuantumGate(n_qubits)
        self.update_q = self._QuantumGate(n_qubits)
        self.output_q = self._QuantumGate(n_qubits)

        # Combiner to fuse classical and quantum outputs
        self.combiner = nn.Linear(2 * hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.shape[1]
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional tuple (hx, cx) of shape
                    (batch, hidden_dim)
        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            (hx, cx): Final hidden and cell states
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for step in range(inputs.shape[0]):
            x = inputs[step]
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = torch.sigmoid(self.forget_linear(combined))
            i_c = torch.sigmoid(self.input_linear(combined))
            g_c = torch.tanh(self.update_linear(combined))
            o_c = torch.sigmoid(self.output_linear(combined))

            # Quantum gate outputs (use first n_qubits of the linear output)
            f_q = torch.sigmoid(self.forget_q(self.forget_linear(combined)[:, :self.n_qubits]))
            i_q = torch.sigmoid(self.input_q(self.input_linear(combined)[:, :self.n_qubits]))
            g_q = self.update_q(self.update_linear(combined)[:, :self.n_qubits])
            o_q = torch.sigmoid(self.output_q(self.output_linear(combined)[:, :self.n_qubits]))

            # Concatenate classical and quantum parts
            f = torch.cat([f_c, f_q], dim=1)
            i = torch.cat([i_c, i_q], dim=1)
            g = torch.cat([g_c, g_q], dim=1)
            o = torch.cat([o_c, o_q], dim=1)

            # Fuse via combiner
            f = torch.sigmoid(self.combiner(f))
            i = torch.sigmoid(self.combiner(i))
            g = torch.tanh(self.combiner(g))
            o = torch.sigmoid(self.combiner(o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class Attention(nn.Module):
    """
    Simple dot‑product attention over the LSTM outputs.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (seq_len, batch, hidden_dim)
        Returns:
            context: Tensor of shape (batch, hidden_dim)
        """
        # Compute query vector
        query = self.query_proj(hidden_states[-1])  # use last hidden state as query
        # Compute attention scores
        scores = torch.matmul(hidden_states.transpose(0, 1), query.unsqueeze(-1)).squeeze(-1)
        attn_weights = F.softmax(scores, dim=1)
        # Weighted sum of hidden states
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states.transpose(0, 1)).squeeze(1)
        return context

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses either :class:`QLSTMGen` or ``nn.LSTM``.
    When a quantum path is enabled, a small attention module follows the
    LSTM to aggregate information across the sequence.
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
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits=n_qubits)
            self.attention = Attention(hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.attention = None
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: Tensor of shape (seq_len, batch)
        Returns:
            log probabilities over tags: (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        if self.attention is not None:
            # Use attention context for each token
            context = self.attention(lstm_out)
            # Broadcast context to all positions
            context = context.unsqueeze(0).expand_as(lstm_out)
            lstm_out = lstm_out + context
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMGen", "LSTMTagger"]
