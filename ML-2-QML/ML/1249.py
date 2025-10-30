import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QuantumFeatureMap(nn.Module):
    """
    Classical approximation of a quantum feature map using random Fourier features.
    The map expands the input into a higher‑dimensional space via sin/cos transforms.
    """
    def __init__(self, input_dim: int, output_dim: int = 16, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        # Fixed random projection (non‑trainable)
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1, requires_grad=False)
        self.b = nn.Parameter(torch.randn(output_dim) * 0.1, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        z = x @ self.W + self.b
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (batch, 2*output_dim)


class QLSTM(nn.Module):
    """
    Hybrid LSTM cell that augments classical linear gates with a quantum‑inspired
    feature map.  The interface matches the original QLSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 feature_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim

        self.feature_map = QuantumFeatureMap(input_dim, feature_dim)
        combined_dim = input_dim + hidden_dim + 2 * feature_dim  # 2* because of sin/cos

        # Classical linear gates
        self.forget_linear = nn.Linear(combined_dim, hidden_dim)
        self.input_linear = nn.Linear(combined_dim, hidden_dim)
        self.update_linear = nn.Linear(combined_dim, hidden_dim)
        self.output_linear = nn.Linear(combined_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            f_map = self.feature_map(x)  # (batch, 2*feature_dim)
            combined = torch.cat([x, hx, f_map], dim=-1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the hybrid QLSTM and a vanilla nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
