"""Hybrid sequence model with optional quantum‑style extensions (classical implementation)."""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Classical kernel and convolution helpers
# --------------------------------------------------------------------------- #
class KernelAnsatz(nn.Module):
    """Radial‑basis function ansatz compatible with the quantum interface."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that exposes a single‑pair kernel call."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = Kernel(gamma)
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])


class ConvFilter(nn.Module):
    """Simple 2‑D convolution layer that emulates a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


# --------------------------------------------------------------------------- #
# 2. Classical LSTM cell (drop‑in quantum replacement)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Pure‑PyTorch LSTM cell mimicking the quantum interface."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        return torch.cat(outputs, dim=0), (hx, cx)

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


# --------------------------------------------------------------------------- #
# 3. Hybrid tagger that can inject kernel / conv features
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Sequence tagger that can operate in classical or quantum mode and optionally
    prepend quantum‑style kernel embeddings and convolutional filters."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
        use_conv: bool = False,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_kernel = use_kernel
        self.use_conv = use_conv
        if use_kernel:
            self.kernel = Kernel(kernel_gamma)
        if use_conv:
            self.conv = ConvFilter(conv_kernel_size, conv_threshold)
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # Embed words
        embeds = self.word_embeddings(sentence)

        # Optionally prepend kernel or conv features
        if self.use_kernel:
            # Pairwise kernel between all time steps (simplified: use first token as anchor)
            anchors = embeds[0].unsqueeze(0).expand(embeds.size(0), -1)
            embeds = torch.cat([embeds, self.kernel(anchors, embeds).unsqueeze(-1)], dim=-1)

        if self.use_conv:
            # Treat each embedding as a 2‑D patch
            conv_out = self.conv(embeds.unsqueeze(1))
            embeds = torch.cat([embeds, conv_out.unsqueeze(-1)], dim=-1)

        # LSTM
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)

        # Tag logits
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


# --------------------------------------------------------------------------- #
# 4. Classical classifier factory (depth‑wise fully‑connected)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, Iterable[int], Iterable[int], list[int]]:
    """Return a feed‑forward network and metadata mimicking the quantum version."""
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = [
    "HybridQLSTM",
    "build_classifier_circuit",
    "Kernel",
    "ConvFilter",
]
