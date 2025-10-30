import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLayer(tq.QuantumModule):
    """
    Parameterized quantum layer that maps an input vector to a probability vector
    using a depth‑controlled circuit.
    """
    def __init__(self, n_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth

        # Linear encoder from input to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)
            ]
        )

        # Parameterized rotation layers
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])

        # Entangling layers repeated `depth` times
        self.entangle = nn.ModuleList([
            tq.CNOT(wires=[i, (i+1)%n_wires]) for i in range(n_wires)
        ])

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for _ in range(self.depth):
            for gate in self.entangle:
                gate(qdev)
        return self.measure(qdev)

class QLSTMGen(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is a QLayer outputting a probability
    vector. The cell is fully differentiable and can be trained end‑to‑end.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget = QLayer(n_qubits, depth=depth)
        self.input = QLayer(n_qubits, depth=depth)
        self.update = QLayer(n_qubits, depth=depth)
        self.output = QLayer(n_qubits, depth=depth)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTaggerGen(nn.Module):
    """
    Sequence tagging model that uses the quantum LSTM cell. It is compatible
    with the classical counterpart via the same name.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int, depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
