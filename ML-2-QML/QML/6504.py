import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLayer(tq.QuantumModule):
    """Small quantum circuit used as a gate."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical inputs to Ry rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        # Trainable single‑qubit rotations
        self.trainable = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        # Entangling layer
        self.entangle = nn.ModuleList([tq.CNOT(has_params=False, trainable=False) for _ in range(n_wires-1)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.trainable):
            gate(qdev, wires=wire)
        for gate in self.entangle:
            gate(qdev, wires=[gate.wires[0], gate.wires[1]])
        return self.measure(qdev)

class QSampler(tq.QuantumModule):
    """Quantum sampler that turns a hidden state into a probability distribution over tags."""
    def __init__(self, tagset_size: int) -> None:
        super().__init__()
        self.tagset_size = tagset_size
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(tagset_size)]
        )
        self.entangle = nn.ModuleList([tq.CNOT(has_params=False, trainable=False) for _ in range(tagset_size-1)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden shape: (batch, hidden_dim)
        angles = hidden[:, :self.tagset_size]
        qdev = tq.QuantumDevice(n_wires=self.tagset_size, bsz=angles.shape[0], device=angles.device)
        self.encoder(qdev, angles)
        for gate in self.entangle:
            gate(qdev, wires=[gate.wires[0], gate.wires[1]])
        probs = self.measure(qdev)
        # Convert expectation values to probabilities
        probs = (probs + 1.0) / 2.0
        return probs

class QLSTMHybrid(nn.Module):
    """Quantum‑enhanced LSTM that can be switched on or off."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.forget = QLayer(n_qubits)
            self.input = QLayer(n_qubits)
            self.update = QLayer(n_qubits)
            self.output = QLayer(n_qubits)
            self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.sampler = QSampler(tagset_size)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.sampler = None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            probs = F.softmax(tag_logits, dim=-1)
        else:
            hx = torch.zeros(embeds.size(1), self.hidden_dim, device=embeds.device)
            cx = torch.zeros(embeds.size(1), self.hidden_dim, device=embeds.device)
            outputs = []
            for x in embeds.unbind(dim=0):
                combined = torch.cat([x, hx], dim=1)
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0)
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            probs = self.sampler(tag_logits)
        return torch.log(probs + 1e-12)

__all__ = ["QLSTMHybrid"]
