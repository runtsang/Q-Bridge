import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLayer(nn.Module):
    """Variational quantum circuit implemented with PennyLane, supports batched inputs."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(0.01 * torch.randn(n_qubits))

        def circuit(x, params):
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
                qml.RX(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.fun = qml.batch(circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fun(x, self.params)

class QLSTM(nn.Module):
    """Quantum‑augmented LSTM that can operate in full quantum mode or fallback to classical gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 use_classical: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_classical = use_classical

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if not self.use_classical:
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

            if not self.use_classical:
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
        if not self.use_classical:
            q_outputs = torch.cat(q_outputs, dim=0)
            return outputs, (h, c), q_outputs
        else:
            return outputs, (h, c)

    def consistency_loss(self, q_out: torch.Tensor, c_out: torch.Tensor) -> torch.Tensor:
        """Mean‑squared‑error between quantum and classical hidden states."""
        return F.mse_loss(q_out, c_out)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can toggle between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, use_classical: bool = False):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                          use_classical=use_classical)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
