from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# QRNN utilities – used only for demonstration
from.QRNN import feedforward_without_memory, random_network

# Classical classifier builder
from.QuantumClassifierModel import build_classifier_circuit

# Fully–connected layer
from.FCL import FCL


class ClassicalQLSTM(nn.Module):
    """Classical imitation of the quantum LSTM gates using linear layers."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class SharedClassName(nn.Module):
    """Hybrid LSTM tagger with optional quantum‑style gates, QRNN utilities,
    and classifier heads."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 classifier_depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Classical classifier circuit (for demonstration)
        self.classifier, self.enc, self.wts, self.obs = build_classifier_circuit(
            num_features=embedding_dim,
            depth=classifier_depth
        )

        self.fcl = FCL()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(1)  # batch_first
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

    def random_qnn_output(self, samples: int = 10):
        """Generate feed‑forward outputs from a random QRNN without memory."""
        arch, unitaries, training_data, target = random_network([2, 4, 2], samples)
        return feedforward_without_memory(training_data, unitaries, arch)

    def fcl_run(self, thetas):
        return self.fcl.run(thetas)


__all__ = ["SharedClassName"]
