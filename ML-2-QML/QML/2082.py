import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM where each gate is a variational multi‑qubit circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Variational parameters per wire
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3)) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            # Encode input values onto qubits
            for i in range(self.n_wires):
                tqf.rx(qdev, x[:, i], wires=[i])
            # Variational rotations
            for i, param in enumerate(self.params):
                tqf.rx(qdev, param[0], wires=[i])
                tqf.ry(qdev, param[1], wires=[i])
                tqf.rz(qdev, param[2], wires=[i])
            # Entanglement
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i+1])
            tqf.cnot(qdev, wires=[self.n_wires-1, 0])
            return self.measure(qdev)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 n_layers: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=bias)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=bias)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=bias)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits, bias=bias)

        # Quantum layers per gate
        self.forget_q = self.QLayer(n_qubits)
        self.input_q = self.QLayer(n_qubits)
        self.update_q = self.QLayer(n_qubits)
        self.output_q = self.QLayer(n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_q(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_q(self.input_lin(combined)))
            g = torch.tanh(self.update_q(self.update_lin(combined)))
            o = torch.sigmoid(self.output_q(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
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
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 n_layers: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              n_layers=n_layers,
                              bias=bias)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                                batch_first=False,
                                bias=bias)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
