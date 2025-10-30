import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullyConnectedLayer(nn.Module):
    """Classic stand‑in for the quantum fully connected layer."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas):
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean()
        return expectation.detach().numpy()

class EstimatorQNN(nn.Module):
    """Classical feed‑forward regressor mimicking the quantum EstimatorQNN."""
    def __init__(self, in_dim: int = 2, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumGateSim(nn.Module):
    """Simulated quantum gate implemented with classical linear layers."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.entangle = nn.Linear(n_qubits, n_qubits, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        x = x * torch.sin(self.params)  # mimic rotation
        x = self.entangle(x)
        return x.mean(dim=1, keepdim=True)

class QuantumLSTMCell(nn.Module):
    """LSTM cell where each gate is a simulated quantum module."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QuantumGateSim(n_qubits)
        self.input_gate = QuantumGateSim(n_qubits)
        self.update_gate = QuantumGateSim(n_qubits)
        self.output_gate = QuantumGateSim(n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.lin_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.lin_input(combined)))
        g = torch.tanh(self.update_gate(self.lin_update(combined)))
        o = torch.sigmoid(self.output_gate(self.lin_output(combined)))

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class UnifiedQLSTM(nn.Module):
    """Hybrid classical‑quantum LSTM tagger with switchable estimator."""
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)

        if use_quantum and n_qubits > 0:
            self.lstm_cell = QuantumLSTMCell(input_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.use_quantum:
            seq_len, batch, _ = embeds.shape
            hx = torch.zeros(batch, self.hidden_dim, device=embeds.device)
            cx = torch.zeros(batch, self.hidden_dim, device=embeds.device)
            outputs = []
            for t in range(seq_len):
                hx, cx = self.lstm_cell(embeds[t], hx, cx)
                outputs.append(hx.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(outputs)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["FullyConnectedLayer", "EstimatorQNN", "QuantumGateSim",
           "QuantumLSTMCell", "UnifiedQLSTM"]
