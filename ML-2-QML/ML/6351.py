import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridGate(nn.Module):
    """
    Classical gate that projects to a qubit‑size vector and then applies a small
    MLP to emulate a quantum activation function.
    """
    def __init__(self, in_dim: int, out_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_qubits)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.linear(x))

class QLSTMGen(nn.Module):
    """
    Classical LSTM cell with hybrid gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = HybridGate(input_dim + hidden_dim, hidden_dim, n_qubits, depth)
        self.input = HybridGate(input_dim + hidden_dim, hidden_dim, n_qubits, depth)
        self.update = HybridGate(input_dim + hidden_dim, hidden_dim, n_qubits, depth)
        self.output = HybridGate(input_dim + hidden_dim, hidden_dim, n_qubits, depth)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTaggerGen(nn.Module):
    """
    Sequence tagging model that can swap between the hybrid LSTM and a standard nn.LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
