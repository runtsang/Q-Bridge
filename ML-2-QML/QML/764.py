import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchquantum as tq

class QLayer(tq.QuantumModule):
    # Small variational quantum circuit that outputs a single qubit expectation value.
    # The circuit consists of RX and RZ rotations followed by a chain of CNOTs.
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        # Trainable parameters for RX and RZ
        self.params = nn.Parameter(torch.randn(n_qubits * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Tensor of shape (batch, n_qubits) containing rotation angles.
        # Returns: Tensor of shape (batch,) with the expectation value of PauliZ
        #          on the first qubit after the circuit.
        batch_size = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch_size, device=device)

        # Encode the input into RX rotations
        for i in range(self.n_qubits):
            tq.RX(x[:, i], wires=i)(qdev)

        # Parametric circuit
        for i in range(self.n_qubits):
            tq.RX(self.params[i], wires=i)(qdev)
            tq.RZ(self.params[self.n_qubits + i], wires=i)(qdev)

        # Entanglement
        for i in range(self.n_qubits - 1):
            tq.CNOT(qdev, wires=[i, i + 1])

        # Measurement of PauliZ on the first qubit
        meas = tq.MeasureAll(qdev)
        out = meas(qdev)
        return out[:, 0]  # (batch,)

class QLSTMGen025(nn.Module):
    # Hybrid LSTM cell that replaces classical gates with quantum‑enhanced gates.
    # The quantum circuit provides a bias that is added to the classical linear
    # transform before activation. A residual connection and a transformer‑style
    # self‑attention block are also included.
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, residual: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.residual = residual

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates
        self.forget_q = QLayer(hidden_dim)
        self.input_q = QLayer(hidden_dim)
        self.update_q = QLayer(hidden_dim)
        self.output_q = QLayer(hidden_dim)

        # Residual linear transform
        if self.residual:
            self.residual_linear = nn.Linear(input_dim, hidden_dim)

        # Self‑attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=False)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical linear outputs
            f_lin = self.forget_linear(combined)
            i_lin = self.input_linear(combined)
            g_lin = self.update_linear(combined)
            o_lin = self.output_linear(combined)

            # Quantum biases
            f_q = self.forget_q(f_lin)
            i_q = self.input_q(i_lin)
            g_q = self.update_q(g_lin)
            o_q = self.output_q(o_lin)

            # Gates with quantum bias
            f = torch.sigmoid(f_lin + f_q)
            i = torch.sigmoid(i_lin + i_q)
            g = torch.tanh(g_lin + g_q)
            o = torch.sigmoid(o_lin + o_q)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.residual:
                hx = hx + self.residual_linear(x)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        # Transformer‑style self‑attention
        attn_output, _ = self.attention(outputs, outputs, outputs)
        outputs = outputs + attn_output

        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTaggerGen025(nn.Module):
    # Sequence tagging model that can switch between the hybrid QLSTMGen025
    # and a standard nn.LSTM. The tagger is unchanged from the original
    # but now references the upgraded LSTM cell.
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen025(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
