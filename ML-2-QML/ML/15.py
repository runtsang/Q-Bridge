import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLayer(nn.Module):
    """Variational quantum circuit used as a gate.  Tries torchquantum, falls back to a linear transform."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.param = nn.Parameter(torch.randn(n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_wires)
        try:
            import torchquantum as tq
            from torchquantum.functional import cnot
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            for i in range(self.n_wires):
                tq.RX(self.param[i])(qdev, wires=i)
            for i in range(self.n_wires - 1):
                cnot(qdev, wires=[i, i+1])
            return tq.MeasureAll(tq.PauliZ)(qdev)
        except Exception:
            return x @ torch.diag(self.param)

class QLSTM(nn.Module):
    """Hybrid LSTM that can run in classical mode or with quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_quantum:
            self.forget_q = QLayer(n_qubits)
            self.input_q = QLayer(n_qubits)
            self.update_q = QLayer(n_qubits)
            self.output_q = QLayer(n_qubits)
            self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        h, c = self._init_states(inputs, states)
        outputs = []
        q_outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)

            f_c = torch.sigmoid(self.forget_linear(combined))
            i_c = torch.sigmoid(self.input_linear(combined))
            g_c = torch.tanh(self.update_linear(combined))
            o_c = torch.sigmoid(self.output_linear(combined))

            if self.use_quantum:
                f_q = torch.sigmoid(self.forget_q(self.forget_proj(combined)))
                i_q = torch.sigmoid(self.input_q(self.input_proj(combined)))
                g_q = torch.tanh(self.update_q(self.update_proj(combined)))
                o_q = torch.sigmoid(self.output_q(self.output_proj(combined)))

                h_q = f_q * c + i_q * g_q
                h = 0.5 * (h + h_q)
                c = f_c * c + i_c * g_c
                q_outputs.append(h_q.unsqueeze(0))
            else:
                h = f_c * c + i_c * g_c
                c = f_c * c + i_c * g_c

            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        if self.use_quantum:
            q_outputs = torch.cat(q_outputs, dim=0)
            return outputs, (h, c), q_outputs
        else:
            return outputs, (h, c)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the hybrid QLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_quantum=use_quantum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
