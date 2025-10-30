import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridGate(nn.Module):
    """
    Lightweight, parameter‑shared gate that mimics a quantum variational circuit.
    It produces a scalar in (0,1) for forget, input, and output gates, or in (-1,1)
    for the update gate.  The module is fully differentiable and uses a small
    neural network to emulate quantum behaviour.
    """
    def __init__(self, in_dim: int, n_params: int = 4):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_params)
        self.fc1 = nn.Linear(n_params, n_params)
        self.fc2 = nn.Linear(n_params, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lin(x)
        z = F.relu(self.fc1(z))
        z = torch.tanh(self.fc2(z))  # (-1,1)
        return z

class HybridLSTMCell(nn.Module):
    """
    Classical LSTM cell that uses HybridGate modules for its four gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._gate_size = input_dim + hidden_dim

        # Gate projections
        self.f_gate = HybridGate(self._gate_size)
        self.i_gate = HybridGate(self._gate_size)
        self.g_gate = HybridGate(self._gate_size)
        self.o_gate = HybridGate(self._gate_size)

        # Linear layers to produce gate inputs
        self.f_lin = nn.Linear(self._gate_size, hidden_dim)
        self.i_lin = nn.Linear(self._gate_size, hidden_dim)
        self.g_lin = nn.Linear(self._gate_size, hidden_dim)
        self.o_lin = nn.Linear(self._gate_size, hidden_dim)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.f_gate(self.f_lin(combined)))
        i = torch.sigmoid(self.i_gate(self.i_lin(combined)))
        g = torch.tanh(self.g_gate(self.g_lin(combined)))
        o = torch.sigmoid(self.o_gate(self.o_lin(combined)))

        new_c = f * cx + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

class HybridLSTM(nn.Module):
    """
    Wrapper around HybridLSTMCell to process a whole sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = HybridLSTMCell(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None):
        batch_size = inputs.size(1)
        device = inputs.device
        if h0 is None:
            h0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.cell.hidden_dim, device=device)

        hx, cx = h0, c0
        outputs = []
        for step in range(inputs.size(0)):
            hx, cx = self.cell(inputs[step], hx, cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model using either the hybrid LSTM or a pure
    PyTorch LSTM.  The interface is identical to the original seed.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, use_hybrid: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if use_hybrid:
            self.lstm = HybridLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden_to_tag(lstm_out)
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridGate", "HybridLSTMCell", "HybridLSTM", "LSTMTagger"]
