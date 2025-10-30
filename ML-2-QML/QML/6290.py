import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QLayer(tq.QuantumModule):
    """Variational quantum layer used for each LSTM gate."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Encode classical inputs into rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTM__gen099(nn.Module):
    """Quantum LSTM cell with a learnable variational circuit per gate."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        # For quantum mode, the hidden dimension is set to the number of qubits
        self.hidden_dim = n_qubits if n_qubits > 0 else hidden_dim

        # Quantum gates for each LSTM gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.residual = nn.Identity()
        self.ln = LayerNorm(self.hidden_dim)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx_candidate = o * torch.tanh(cx)
            hx = self.residual(hx_candidate)
            hx = self.ln(self.dropout(hx))
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger__gen099(nn.Module):
    """Sequence tagging model that uses either the extended QLSTM or a standard nn.LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM__gen099(embedding_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out = lstm_out.squeeze(1)
        scores = torch.softmax(torch.matmul(lstm_out, lstm_out.transpose(0, 1)), dim=-1)
        context = torch.matmul(scores, lstm_out)
        tag_logits = self.hidden2tag(context)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen099", "LSTMTagger__gen099"]
