import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is a variational quantum circuit.
    The circuit is defined by a ladder of RX, RZ, and CNOT gates.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Parameterised RX and RZ gates
            self.rz_params = nn.Parameter(torch.randn(n_wires))
            self.rx_params = nn.Parameter(torch.randn(n_wires))
            # Entanglement parameters for CNOT ladder
            self.cnot_params = nn.Parameter(torch.randn(n_wires - 1))
            # Encoder to map classical input to qubit states
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]}]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape (batch, 1)
            batch_size = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device)
            self.encoder(qdev, x)
            for i in range(self.n_wires):
                tq.RX(self.rx_params[i], wires=i, qdev=qdev)
                tq.RZ(self.rz_params[i], wires=i, qdev=qdev)
            for i in range(self.n_wires - 1):
                tq.CNOT(qdev, wires=[i, i + 1])
            z = tqf.expectation(qdev, tq.PauliZ, wires=range(self.n_wires))
            return z.mean(-1, keepdim=True)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.lin_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_g = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_o = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        # inputs shape (batch, seq_len, feature)
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_f(combined)))
            i = torch.sigmoid(self.input(self.lin_i(combined)))
            g = torch.tanh(self.update(self.lin_g(combined)))
            o = torch.sigmoid(self.output(self.lin_o(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between quantum and classical LSTM.
    Includes a gated attention mechanism over the LSTM outputs.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence shape (batch, seq_len)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        tag_logits = self.hidden2tag(context)
        return F.log_softmax(tag_logits, dim=-1)
