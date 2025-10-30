import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Import the quantum gate layer from the QML module
try:
    from qml_code import QLayerQuantum
except ImportError:
    # Fallback dummy implementation if the quantum module is not available
    class QLayerQuantum(nn.Module):
        def __init__(self, n_qubits: int, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

class QLSTM_Enhanced(nn.Module):
    """
    Hybrid LSTM cell with optional quantum gates.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        gate_mix: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_mix = nn.Parameter(torch.tensor(gate_mix, dtype=torch.float32))

        # Classical linear gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates
        if n_qubits > 0:
            self.forget_q = QLayerQuantum(n_qubits, hidden_dim)
            self.input_q  = QLayerQuantum(n_qubits, hidden_dim)
            self.update_q = QLayerQuantum(n_qubits, hidden_dim)
            self.output_q = QLayerQuantum(n_qubits, hidden_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = torch.sigmoid(self.forget_lin(combined))
            i_c = torch.sigmoid(self.input_lin(combined))
            g_c = torch.tanh(self.update_lin(combined))
            o_c = torch.sigmoid(self.output_lin(combined))

            if self.n_qubits > 0:
                # Quantum gate outputs
                f_q = torch.sigmoid(self.forget_q(combined))
                i_q = torch.sigmoid(self.input_q(combined))
                g_q = torch.tanh(self.update_q(combined))
                o_q = torch.sigmoid(self.output_q(combined))

                # Weighted mix
                f = self.gate_mix * f_q + (1 - self.gate_mix) * f_c
                i = self.gate_mix * i_q + (1 - self.gate_mix) * i_c
                g = self.gate_mix * g_q + (1 - self.gate_mix) * g_c
                o = self.gate_mix * o_q + (1 - self.gate_mix) * o_c
            else:
                f, i, g, o = f_c, i_c, g_c, o_c

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    Sequence tagging model that can switch between classical and hybrid LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        gate_mix: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM_Enhanced(embedding_dim, hidden_dim, n_qubits=n_qubits, gate_mix=gate_mix)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM_Enhanced", "LSTMTagger"]
