import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell.  Each gate is realised by a small variational
    circuit with a user‑controlled depth.  The rest of the architecture mirrors
    the classical counterpart so that the tagger can swap seamlessly.
    """
    class QLayer(nn.Module):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameters for each rotation gate
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires, 3)) for _ in range(depth)]
            )
            self.dev = qml.device("default.qubit", wires=n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            @qml.qnode(self.dev, interface="torch")
            def circuit(data):
                # Encode data into first wire
                qml.RX(data[0], wires=0)
                # Variational layers
                for i, p in enumerate(self.params):
                    for w in range(self.n_wires):
                        qml.Rot(p[w, 0], p[w, 1], p[w, 2], wires=w)
                    for w in range(self.n_wires - 1):
                        qml.CNOT(wires=[w, w + 1])
                return qml.expval(qml.PauliZ(0))
            return circuit(x)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 depth: int = 2, dropout_prob: float = 0.0, attn_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None
        self.attn_dim = attn_dim

        # Linear projections to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_q = self.QLayer(n_qubits, depth)
        self.input_q = self.QLayer(n_qubits, depth)
        self.update_q = self.QLayer(n_qubits, depth)
        self.output_q = self.QLayer(n_qubits, depth)

        # Attention
        if self.attn_dim > 0:
            self.attn_proj = nn.Linear(hidden_dim, attn_dim)
            self.attn_score = nn.Linear(attn_dim, 1, bias=False)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_q(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_q(self.linear_input(combined)))
            g = torch.tanh(self.update_q(self.linear_update(combined)))
            o = torch.sigmoid(self.output_q(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.dropout is not None:
                hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def forward_with_attention(self, inputs: torch.Tensor,
                               states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        lstm_out, (hx, cx) = self.forward(inputs, states)
        if self.attn_dim == 0:
            return lstm_out, (hx, cx)
        attn_h = torch.tanh(self.attn_proj(lstm_out))
        attn_weights = F.softmax(self.attn_score(attn_h).squeeze(-1), dim=0)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
        return context, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum‑enhanced LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int, depth: int = 2,
                 dropout_prob: float = 0.0, attn_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                          depth=depth, dropout_prob=dropout_prob,
                          attn_dim=attn_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def forward_with_attention(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        context, _ = self.lstm.forward_with_attention(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(context.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
