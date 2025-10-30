import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QLSTM(nn.Module):
    """
    Quantum LSTM cell with variational quantum gates and residual skip connections.
    The quantum gates are realized by a small parameterized circuit per gate.
    """
    class QGate(tq.QuantumModule):
        """
        Variational gate that maps a classical vector to a quantum state.
        The circuit consists of:
            - RX encoding of input features,
            - Parameterized RX, RY, RZ rotations,
            - Entanglement via a fixed CNOT chain.
        The output is the vector of Pauli‑Z expectation values of each qubit.
        """
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.params = nn.Parameter(torch.randn(n_wires, 3))
            self.entanglement = [(i, i + 1) for i in range(n_wires - 1)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            for w in range(self.n_wires):
                val = x[:, w % x.shape[1]] if x.shape[1] > w else 0
                tq.RX(val)(qdev, wires=w)
            for w in range(self.n_wires):
                tq.RX(self.params[w, 0])(qdev, wires=w)
                tq.RY(self.params[w, 1])(qdev, wires=w)
                tq.RZ(self.params[w, 2])(qdev, wires=w)
            for a, b in self.entanglement:
                tq.CNOT(qdev, wires=[a, b])
            return tq.MeasureAll(tq.PauliZ)(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 dropout: float = 0.1, residual: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=True)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=True)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=True)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx_new = o * torch.tanh(cx)
            if self.residual:
                hx_new = hx_new + hx
            hx_new = self.layernorm(hx_new)
            hx_new = self.dropout(hx_new)
            hx = hx_new
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits,
                dropout=dropout, residual=residual
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
