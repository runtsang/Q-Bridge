import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class _VariationalQuantumCircuit(nn.Module):
    """Lightweight variational circuit implemented with classical linear layers.
    Mimics the role of a quantum circuit in the classical baseline.
    """
    def __init__(self, hidden_dim: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        for layer in self.layers:
            weight_norm(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

    def orthogonality_penalty(self) -> torch.Tensor:
        penalty = 0.0
        for layer in self.layers:
            w = layer.weight
            wt_w = w @ w.t()
            penalty += torch.norm(wt_w - torch.eye(wt_w.size(0), device=w.device), p='fro') ** 2
        return penalty

class QLSTM(nn.Module):
    """Classical LSTM with optional variational quantum‑style circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 depth: int = 1, orthogonal_reg: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.orthogonal_reg = orthogonal_reg

        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.qc = _VariationalQuantumCircuit(hidden_dim, depth)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            f = self.qc(f)
            i = self.qc(i)
            g = self.qc(g)
            o = self.qc(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

    def orthogonality_loss(self) -> torch.Tensor:
        if self.orthogonal_reg == 0:
            return torch.tensor(0.0, device=self.qc.layers[0].weight.device)
        return self.orthogonal_reg * self.qc.orthogonality_penalty()

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1,
                 orthogonal_reg: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits,
                              depth=depth, orthogonal_reg=orthogonal_reg)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
