import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QuantumLSTM(nn.Module):
    """Classic LSTM cell augmented with a variational quantum gate."""
    class _QuantumGate(nn.Module):
        """Variational gate with mean‑field or entangled mode."""
        def __init__(self, n_qubits: int, depth: int = 1, mean_field: bool = True):
            super().__init__()
            self.n_qubits = n_qubits
            self.depth = depth
            self.mean_field = mean_field
            if self.mean_field:
                self.rotations = nn.Parameter(torch.randn(n_qubits))
            else:
                self.rotations = nn.ParameterList(
                    [nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)]
                )
                self.interaction = nn.Parameter(torch.randn(n_qubits, n_qubits))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.mean_field:
                return torch.sigmoid(x + self.rotations)
            else:
                out = x
                for rot in self.rotations:
                    out = torch.sigmoid(out + rot)
                    out = torch.sigmoid(out @ self.interaction)
                return out

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 depth: int = 1,
                 mean_field: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.mean_field = mean_field

        self.forget_gate = self._QuantumGate(n_qubits, depth, mean_field)
        self.input_gate = self._QuantumGate(n_qubits, depth, mean_field)
        self.update_gate = self._QuantumGate(n_qubits, depth, mean_field)
        self.output_gate = self._QuantumGate(n_qubits, depth, mean_field)

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(self.forget_linear(combined))
            i = self.input_gate(self.input_linear(combined))
            g = self.update_gate(self.update_linear(combined))
            o = self.output_gate(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model capable of using a classical or quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 depth: int = 1,
                 mean_field: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumLSTM(embedding_dim,
                                    hidden_dim,
                                    n_qubits,
                                    depth=depth,
                                    mean_field=mean_field)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QuantumLSTM", "LSTMTagger"]
