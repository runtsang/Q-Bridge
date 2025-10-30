import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class _ClassicalGate(nn.Module):
    """Linear gate with sigmoid or tanh activation."""
    def __init__(self, in_dim: int, out_dim: int, activation: str = "sigmoid"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class _QuantumGate(nn.Module):
    """Variational quantum circuit returning a vector of length n_qubits."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.qnode = qml.qnode(self._circuit, interface="torch", device=self.device)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.n_qubits)
        out = []
        for vec in flat:
            out.append(self.qnode(vec, self.params))
        out = torch.stack(out, dim=0)
        out = out.reshape(*batch_shape, self.n_qubits)
        return out

class QLSTM(nn.Module):
    """Hybrid classical‑quantum LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        in_dim = input_dim + hidden_dim
        self.forget_gate = _ClassicalGate(in_dim, hidden_dim, "sigmoid")
        self.input_gate = _ClassicalGate(in_dim, hidden_dim, "sigmoid")
        self.update_gate = _ClassicalGate(in_dim, hidden_dim, "tanh")
        self.output_gate = _ClassicalGate(in_dim, hidden_dim, "sigmoid")
        self.quantum_gate = _QuantumGate(n_qubits) if n_qubits > 0 else None
        self.reg_loss = 0.0

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(combined)
            i = self.input_gate(combined)
            g = self.update_gate(combined)
            o = self.output_gate(combined)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.quantum_gate is not None:
                q_out = self.quantum_gate(combined)
                hx = hx + q_out
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        if self.quantum_gate is not None:
            prob = (q_out + 1.0) / 2.0
            kl = torch.sum(
                prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8),
                dim=-1,
            )
            self.reg_loss = torch.mean(kl).detach()
        else:
            self.reg_loss = 0.0
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
    """Sequence tagging model with optional quantum LSTM and multi‑head attention."""
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
        self.n_heads = n_heads
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

    @property
    def reg_loss(self) -> torch.Tensor:
        return getattr(self.lstm, "reg_loss", torch.tensor(0.0))

__all__ = ["QLSTM", "LSTMTagger"]
