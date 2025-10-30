import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pennylane as qml
from pennylane import numpy as np

class QLayer(nn.Module):
    """
    Variational quantum layer using Pennylane.
    Maps an input vector of length n_wires to an output vector of the same size.
    """
    def __init__(self, n_wires: int, n_layers: int = 2, device: str = "cpu"):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires, shots=0)
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(x, params):
            for i in range(n_wires):
                qml.RY(x[i], wires=i)
            for l in range(n_layers):
                for w in range(n_wires):
                    qml.RY(params[l, w, 0], wires=w)
                    qml.RZ(params[l, w, 1], wires=w)
                    qml.RX(params[l, w, 2], wires=w)
                for w in range(n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[n_wires - 1, 0])
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_wires)
        returns: (batch_size, n_wires)
        """
        return self.circuit(x, self.params)

class QLSTM(nn.Module):
    """
    Hybrid classical‑quantum LSTM cell.
    Gates are computed from the sum of classical linear outputs and quantum
    circuit outputs.  The quantum part is a variational circuit that
    produces gate logits.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 8, dropout: float = 0.0, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = device

        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.qgate = QLayer(n_qubits, device=device)
        self.dropout = nn.Dropout(dropout)

    def _init_states(self, batch_size: int, device: torch.device):
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self,
                inputs: torch.Tensor,
                seq_lengths: torch.Tensor | None = None,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        inputs: (seq_len, batch, input_dim)
        seq_lengths: optional tensor of lengths for packed sequences
        states: optional (hx, cx)
        """
        device = inputs.device
        batch_size = inputs.size(1)

        if states is None:
            hx, cx = self._init_states(batch_size, device)
        else:
            hx, cx = states

        outputs = []
        for t in range(inputs.size(0)):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)
            f_lin = self.lin_forget(combined)
            i_lin = self.lin_input(combined)
            g_lin = self.lin_update(combined)
            o_lin = self.lin_output(combined)

            f_q = self.qgate(f_lin)
            i_q = self.qgate(i_lin)
            g_q = self.qgate(g_lin)
            o_q = self.qgate(o_lin)

            f = torch.sigmoid(f_lin + f_q)
            i = torch.sigmoid(i_lin + i_q)
            g = torch.tanh(g_lin + g_q)
            o = torch.sigmoid(o_lin + o_q)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical nn.LSTM
    and the hybrid QLSTM.  Supports packed sequences and dropout.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 dropout: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              dropout=dropout,
                              device=device)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                sentences: torch.Tensor,
                seq_lengths: torch.Tensor | None = None):
        """
        sentences: (batch, seq_len)
        seq_lengths: optional tensor of lengths for packing
        """
        embeds = self.word_embeddings(sentences)
        if seq_lengths is not None:
            packed = pack_padded_sequence(
                embeds, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            if isinstance(self.lstm, nn.LSTM):
                packed_out, _ = self.lstm(packed)
                out, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                packed_out, _ = self.lstm(packed.data.transpose(0,1), None)
                out = packed_out.transpose(0,1)
                out = torch.nn.functional.pad(out,
                                               (0,0,0,embeds.size(1)-out.size(1)))
        else:
            if isinstance(self.lstm, nn.LSTM):
                out, _ = self.lstm(embeds)
            else:
                out, _ = self.lstm(embeds.transpose(0,1), None)
                out = out.transpose(0,1)
        logits = self.hidden2tag(self.dropout(out))
        return F.log_softmax(logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
