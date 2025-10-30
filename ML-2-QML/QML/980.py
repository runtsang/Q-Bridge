import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell.

    Features:
    * 1‑D convolutional encoder for local context
    * Variational quantum circuit (parameterized rotations + entangling CNOTs)
    * Dropout applied to quantum output
    * Residual connection from hidden state
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Parameterized rotation gates
            self.params = nn.Parameter(torch.randn(n_wires, 3))
            # Entangling pattern
            self.cnot_pattern = [(i, i + 1) for i in range(n_wires - 1)] + [(n_wires - 1, 0)]

        def forward(self, x: Tensor) -> Tensor:
            # x shape: (batch, n_wires)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            # Encode classical data into qubits via RX rotations
            for i in range(self.n_wires):
                tqf.rx(qdev, wires=[i], theta=x[:, i])
            # Apply trainable rotations
            for i in range(self.n_wires):
                theta = self.params[i]
                tqf.rx(qdev, wires=[i], theta=theta[0])
                tqf.ry(qdev, wires=[i], theta=theta[1])
                tqf.rz(qdev, wires=[i], theta=theta[2])
            # Entangling CNOTs
            for control, target in self.cnot_pattern:
                tqf.cnot(qdev, wires=[control, target])
            # Measure all qubits in Z basis
            return tqf.measure_all(qdev, basis='z')

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout_prob: float = 0.1,
        conv_kernel: int = 3,
        conv_channels: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout_prob = dropout_prob

        # Classical encoder
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
        )
        self.bn = nn.BatchNorm1d(conv_channels)

        # Linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate
        self.qgate = self.QLayer(n_qubits)

        # Dropout on quantum output
        self.q_dropout = nn.Dropout(p=dropout_prob)

    def _init_states(self, inputs: Tensor, states: Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.shape[1]
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: Tensor,
        states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            # Convolutional encoding
            x_enc = self.conv(x.unsqueeze(-1))
            x_enc = self.bn(x_enc)
            x_enc = torch.relu(x_enc).squeeze(-1)

            combined = torch.cat([x_enc, hx], dim=1)

            # Classical gates
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # Quantum contribution
            q_out = self.qgate(combined)
            q_out = self.q_dropout(q_out)

            # Update cell state
            cx = f * cx + i * (g + q_out)

            # Residual hidden update
            hx = o * torch.tanh(cx) + 0.1 * hx

            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum‑enhanced LSTM cell.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            dropout_prob=dropout_prob,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: Tensor) -> Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
