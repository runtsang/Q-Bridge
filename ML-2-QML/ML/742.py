import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Classical LSTM cell that mimics the interface of the quantum variant.
    The gates are implemented with standard linear layers and sigmoid/tanh
    activations.  This class is kept for backward compatibility and to
    provide a drop‑in replacement when a quantum backend is not desired.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # unused but kept for API compatibility
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class HybridQLSTM(nn.Module):
    """
    Wrapper that can switch between a classical LSTM and the quantum
    variant at runtime.  The flag ``use_quantum`` controls the choice.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        if use_quantum:
            # Lazy import to avoid circular dependency
            from.qml import QLSTM as QuantumQLSTM

            self.lstm = QuantumQLSTM(input_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_quantum:
            return self.lstm(inputs, states)
        else:
            return self.lstm(inputs, states)

    def evaluate_on_backend(self, backend_name: str = "default") -> str:
        """
        Convenience wrapper that forwards the call to the quantum LSTM
        when ``use_quantum`` is True.  For the classical case a
        RuntimeError is raised.
        """
        if self.use_quantum:
            return self.lstm.evaluate_on_backend(backend_name)
        raise RuntimeError("No quantum backend available for classical LSTM.")

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can operate with either the classical
    or the quantum LSTM cell.  The ``use_quantum`` flag selects the mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, use_quantum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "HybridQLSTM", "LSTMTagger"]
