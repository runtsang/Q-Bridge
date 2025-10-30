"""QLSTM__gen007.py – Classical side of the hybrid LSTM tagger.

This module contains:
* A fully classical LSTM implementation (QLSTM) that mirrors the quantum gate
  interface.
* A hybrid activation layer (HybridFunction / Hybrid) that can be swapped
  for a quantum expectation head.
* Convenience wrappers (`SamplerQNN` and `FCL`) that provide classical
  stand‑ins for the quantum sampler and fully‑connected layer used in the
  QML side.
* `LSTMTagger`, the main model, which can use either the classical or
  quantum LSTM cell and can optionally replace the final linear layer with
  a quantum hybrid head.

The API is identical to the original `QLSTM.py` so existing training
scripts continue to work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
# Classical LSTM cell – identical API to the quantum version
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM cell.

    The gates are simple linear layers followed by the usual sigmoid/tanh
    activations.  The class signature matches the quantum implementation
    so that `LSTMTagger` can switch between them via the `n_qubits` flag.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
# Hybrid activation – quantum expectation head
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that mimics a quantum expectation value."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Linear head followed by the `HybridFunction`."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --------------------------------------------------------------------------- #
# Classical stand‑ins for quantum modules
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    """Simple softmax network that mimics the quantum sampler."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


def FCL() -> nn.Module:
    """Fully connected layer that returns a single expectation‑like value."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


# --------------------------------------------------------------------------- #
# Main tagger – can use classical or quantum LSTM and an optional quantum head
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that supports classical, quantum, and hybrid heads.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        If > 0, a quantum LSTM cell is used.
    use_quantum_head : bool, default False
        If True, replace the final linear layer with a `Hybrid` head.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_head: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        if use_quantum_head:
            self.head = Hybrid(hidden_dim, shift=np.pi / 2)
        else:
            self.head = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # LSTM expects (seq_len, batch, input_size)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        logits = self.head(lstm_out.squeeze(1))
        if isinstance(self.head, nn.Linear):
            return F.log_softmax(logits, dim=1)
        else:
            # Hybrid head already returns probabilities
            return torch.cat((logits, 1 - logits), dim=-1)


__all__ = [
    "QLSTM",
    "HybridFunction",
    "Hybrid",
    "SamplerQNN",
    "FCL",
    "LSTMTagger",
]
