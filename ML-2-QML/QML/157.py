import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class _QuantumGate(nn.Module):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.device = qml.device("default.qubit", wires=n_wires)
        self.params = nn.Parameter(torch.randn(n_wires))
        self.qnode = qml.qnode(self._circuit, interface="torch", device=self.device)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)
        for i in range(self.n_wires):
            qml.RY(params[i], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.n_wires)
        out = []
        for vec in flat:
            out.append(self.qnode(vec, self.params))
        out = torch.stack(out, dim=0)
        out = out.reshape(*batch_shape, self.n_wires)
        return out

class QLSTM(nn.Module):
    """Pure quantum LSTM cell using PennyLane."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits or hidden_dim
        self.n_wires = input_dim + hidden_dim
        self.forget_gate = _QuantumGate(self.n_wires)
        self.input_gate = _QuantumGate(self.n_wires)
        self.update_gate = _QuantumGate(self.n_wires)
        self.output_gate = _QuantumGate(self.n_wires)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the pure quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        scores = self.attn_score(lstm_out).squeeze(-1)
        attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)
        context = torch.sum(attn_weights * lstm_out, dim=0)
        final = (context + lstm_out[-1]) / 2.0
        tag_logits = self.hidden2tag(final.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
