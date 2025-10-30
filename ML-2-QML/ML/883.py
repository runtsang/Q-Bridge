import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ClassicalGate(nn.Module):
    """Fast classical gate that mimics the sigmoid/tanh behaviour of an LSTM gate."""
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class QuantumEmbedding(nn.Module):
    """Variational encoder that maps token embeddings to a low‑dimensional quantum space."""
    def __init__(self, embed_dim: int, qdim: int, n_wires: int):
        super().__init__()
        self.qdim = qdim
        self.n_wires = n_wires
        self.encoder = nn.Linear(embed_dim, qdim, bias=False)
        self.param_circuit = nn.ModuleList([nn.Linear(qdim, 1) for _ in range(n_wires)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        for i, gate in enumerate(self.param_circuit):
            z[:, i] = torch.tanh(gate(z[:, i:i+1]).squeeze(-1))
        return z

class HybridQLSTM(nn.Module):
    """Hybrid LSTM that uses a quantum‑aware gate for the forget and input gates,
    while keeping the update and output gates classical for speed."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, qdim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.qdim = qdim

        # Quantum‑aware gates
        self.forget_qgate = QuantumEmbedding(input_dim + hidden_dim, qdim, n_qubits)
        self.input_qgate = QuantumEmbedding(input_dim + hidden_dim, qdim, n_qubits)

        # Classical gates
        self.update_gate = ClassicalGate(input_dim + hidden_dim, hidden_dim)
        self.output_gate = ClassicalGate(input_dim + hidden_dim, hidden_dim)

        # Linear projections to match dimensions
        self.proj_to_q = nn.Linear(input_dim + hidden_dim, qdim)
        self.proj_to_hidden = nn.Linear(qdim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_qgate(self.proj_to_q(combined)))
            i = torch.sigmoid(self.input_qgate(self.proj_to_q(combined)))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.shape[1]
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class Tagger(nn.Module):
    """Sequence tagger that uses HybridQLSTM for quantum‑enhanced sequence modelling."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int, qdim: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, qdim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)

__all__ = ["HybridQLSTM", "Tagger"]
