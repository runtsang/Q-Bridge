import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QGate(tq.QuantumModule):
    """A parameterized quantum gate for LSTM gates."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.params = nn.Parameter(torch.randn(n_wires, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.size(0), device=x.device)
        for w in range(self.n_wires):
            tqf.rx(qdev, wires=[w], params=x[:, w].unsqueeze(1))
        for w in range(self.n_wires):
            tqf.rx(qdev, wires=[w], params=self.params[w, 0].unsqueeze(0))
            tqf.ry(qdev, wires=[w], params=self.params[w, 1].unsqueeze(0))
            tqf.rz(qdev, wires=[w], params=self.params[w, 2].unsqueeze(0))
        for src, dst in [(i, (i+1)%self.n_wires) for i in range(self.n_wires)]:
            tqf.cnot(qdev, wires=[src, dst])
        out = []
        for w in range(self.n_wires):
            out.append(tqf.expectation(qdev, tq.PauliZ, wires=[w]))
        return torch.stack(out, dim=1)

class QuantumQLSTMCell(tq.QuantumModule):
    """Quantum‑enhanced LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.qgate_forget = QGate(n_qubits)
        self.qgate_input = QGate(n_qubits)
        self.qgate_update = QGate(n_qubits)
        self.qgate_output = QGate(n_qubits)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        f = torch.sigmoid(self.qgate_forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.qgate_input(self.linear_input(combined)))
        g = torch.tanh(self.qgate_update(self.linear_update(combined)))
        o = torch.sigmoid(self.qgate_output(self.linear_output(combined)))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

class QuantumQLSTM(tq.QuantumModule):
    """Wrapper to process sequences with QuantumQLSTMCell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.cell = QuantumQLSTMCell(input_dim, hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden is None:
            batch_size = inputs.size(1)
            device = inputs.device
            h = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.cell.hidden_dim, device=device)
            hidden = (h, c)
        outputs = []
        for t in range(inputs.size(0)):
            x_t = inputs[t]
            h, hidden = self.cell(x_t, hidden)
            outputs.append(h.unsqueeze(0))
        return torch.cat(outputs, dim=0), hidden

class QuantumLSTMTagger(tq.QuantumModule):
    """Sequence tagging with optional quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        if isinstance(self.lstm, QuantumQLSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=2)
