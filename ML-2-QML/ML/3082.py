import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuantumFullyConnected(nn.Module):
    """
    Classical surrogate of a parameterised quantum fully‑connected layer.
    Uses a linear transform followed by a tanh activation to mimic
    expectation‑value readout of a simple quantum circuit.
    """
    def __init__(self, in_features: int, out_features: int = 1, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class QuantumGateModule(nn.Module):
    """
    Tiny neural network that imitates a single‑qubit variational gate.
    The network takes the concatenated state vector and outputs a
    rotation angle that would be applied to a qubit in the quantum
    version.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QLSTMCell(nn.Module):
    """
    Classical LSTM cell where each gate is computed by a `QuantumGateModule`.
    This mirrors the structure of the quantum LSTM but keeps the entire
    computation on the CPU/GPU for ease of differentiation.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_input = input_dim + hidden_dim

        self.forget_gate = QuantumGateModule(gate_input)
        self.input_gate  = QuantumGateModule(gate_input)
        self.update_gate = QuantumGateModule(gate_input)
        self.output_gate = QuantumGateModule(gate_input)

        self.linear_forget = nn.Linear(gate_input, hidden_dim)
        self.linear_input  = nn.Linear(gate_input, hidden_dim)
        self.linear_update = nn.Linear(gate_input, hidden_dim)
        self.linear_output = nn.Linear(gate_input, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
        g = torch.tanh(self.update_gate(self.linear_update(combined)))
        o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class HybridQLSTM(nn.Module):
    """
    Stacked LSTM that can switch between a classical LSTM and the
    quantum‑gate‑enhanced LSTM cell. The default is the classical
    implementation for speed; set `use_quantum=True` to enable
    the quantum‑style gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            self.layers = nn.ModuleList(
                [QLSTMCell(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [nn.LSTMCell(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)]
            )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = seq.size()
        h = [torch.zeros(batch_size, self.layers[0].hidden_dim, device=seq.device) for _ in self.layers]
        c = [torch.zeros(batch_size, self.layers[0].hidden_dim, device=seq.device) for _ in self.layers]
        outputs = []
        for t in range(seq_len):
            x = seq[:, t, :]
            for i, layer in enumerate(self.layers):
                if self.use_quantum:
                    h[i], c[i] = layer(x, h[i], c[i])
                    x = h[i]
                else:
                    h[i], c[i] = layer(x, (h[i], c[i]))
                    x = h[i]
            outputs.append(x.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class UnifiedQLSTM_FCL(nn.Module):
    """
    End‑to‑end model that embeds tokens, feeds them through a
    quantum‑enhanced LSTM, and finally projects to tag logits via
    a quantum‑style fully‑connected layer.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, tagset_size: int,
                 n_qubits: int = 0, use_quantum_lstm: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = HybridQLSTM(embed_dim, hidden_dim, num_layers=1, use_quantum=use_quantum_lstm)
        self.fc = QuantumFullyConnected(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumFullyConnected", "QLSTMCell", "HybridQLSTM", "UnifiedQLSTM_FCL"]
