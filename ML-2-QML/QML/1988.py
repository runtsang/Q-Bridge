import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLayer(tq.QuantumModule):
    """Variational quantum layer with configurable depth."""
    def __init__(self, n_wires: int, depth: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.rz_params = nn.Parameter(torch.randn(n_wires))
        self.rx_params = nn.Parameter(torch.randn(n_wires))
        self.ry_params = nn.Parameter(torch.randn(n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )(qdev, x)
        for _ in range(self.depth):
            for wire in range(self.n_wires):
                tqf.rx(qdev, wires=wire, params=self.rx_params[wire])
                tqf.ry(qdev, wires=wire, params=self.ry_params[wire])
                tqf.rz(qdev, wires=wire, params=self.rz_params[wire])
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return tq.MeasureAll(tq.PauliZ)(qdev)

class QLSTM(nn.Module):
    """Quantum-enhanced LSTM with variational gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_qgate = QLayer(n_qubits, depth)
        self.input_qgate = QLayer(n_qubits, depth)
        self.update_qgate = QLayer(n_qubits, depth)
        self.output_qgate = QLayer(n_qubits, depth)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined) + self.forget_qgate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_linear(combined) + self.input_qgate(self.input_linear(combined)))
            g = torch.tanh(self.update_linear(combined) + self.update_qgate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_linear(combined) + self.output_qgate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Tagger that switches between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
