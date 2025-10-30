import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, MultiheadAttention

class QuantumGate(nn.Module):
    """Small variational circuit that outputs a single probability value."""
    def __init__(self, n_qubits: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(1)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode input
            for q in range(self.n_qubits):
                qml.RX(x[q], wires=q)
            # Variational layers
            for i in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(params[i, q, 0], wires=q)
                    qml.RZ(params[i, q, 1], wires=q)
                # Entangling layer
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, n_qubits)
        Returns:
            Tensor of shape (batch,) with values in (0,1)
        """
        probs = self.circuit(x, self.params)
        probs = torch.sigmoid(probs)
        probs = self.dropout(probs)
        probs = self.norm(probs.unsqueeze(1)).squeeze(1)
        return probs

class QLSTM(nn.Module):
    """
    Hybrid quantum‑classical LSTM where each gate is realised by a
    tiny variational circuit.  The output of the circuit is combined
    with a classical linear projection to form the gate activations.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 n_layers: int,
                 n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projection to the quantum subspace
        self.proj = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QuantumGate(n_qubits, n_layers, dropout)
        self.input_gate = QuantumGate(n_qubits, n_layers, dropout)
        self.update_gate = QuantumGate(n_qubits, n_layers, dropout)
        self.output_gate = QuantumGate(n_qubits, n_layers, dropout)

        self.attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional previous (hx, cx)
        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            (hx, cx): Final hidden and cell states
        """
        batch_size = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        if states is not None:
            hx, cx = states

        outputs = []
        for t in range(inputs.size(0)):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)
            q_input = self.proj(combined)  # (batch, n_qubits)

            f = torch.sigmoid(self.forget_gate(q_input))
            i = torch.sigmoid(self.input_gate(q_input))
            g = torch.tanh(self.update_gate(q_input))
            o = torch.sigmoid(self.output_gate(q_input))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            if outputs:
                past = torch.stack(outputs, dim=1)  # (batch, t, hidden)
                attn_output, _ = self.attention(
                    query=hx.unsqueeze(1),
                    key=past,
                    value=past,
                )
                hx = hx + attn_output.squeeze(1)

            outputs.append(hx)

        stacked = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the quantum‑enhanced QLSTM
    or a standard nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 n_layers: int = 1,
                 n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              n_qubits=n_qubits,
                              n_layers=n_layers,
                              n_heads=n_heads,
                              dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: Tensor of shape (seq_len, batch)
        Returns:
            Log‑probabilities over tags: (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
