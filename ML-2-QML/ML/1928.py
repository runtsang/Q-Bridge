import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMExtended(nn.Module):
    """Bidirectional, depth‑controlled quantum‑enhanced LSTM.

    The architecture is a drop‑in replacement for the original QLSTM class.  It
    offers three key extensions:
    1. **Bidirectional processing** – optional ``bidirectional=True`` flag.
    2. **Depth control** – each gate’s quantum circuit can be depth‑described
       by a ``depth`` parameter that repeats the basic block.
    3. **Multi‑layer quantum gates** – *M*‑layer per gate, each layer
       `x`‑th layer has an independent set of quantum parameters.
    The class also provides a **pre‑trained** loading helper.
    """
    class _QuantumGate(nn.Module):
        def __init__(self, n_wires: int, depth: int = 0):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.rxs = nn.ModuleList([nn.Linear(n_wires, n_wires) for _ in range(depth)])
            self.measure = nn.Identity()

        def forward(self, x: torch.Tensor):
            # simple linear emulation of a quantum layer; depth controls weight reuse
            out = x
            for layer in self.rxs:
                out = torch.relu(layer(out))
            return out

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        bidirectional: bool = False,
        depth: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.bidirectional = bidirectional
        self.depth = depth

        self.forget = self._QuantumGate(n_qubits, depth) if n_qubits else nn.Identity()
        self.input = self._QuantumGate(n_qubits, depth) if n_qubits else nn.Identity()
        self.update = self._QuantumGate(n_qubits, depth) if n_qubits else nn.Identity()
        self.output = self._QuantumGate(n_qubits, depth) if n_qubits else nn.Identity()

        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if bidirectional:
            self.back_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        if self.bidirectional:
            rev = torch.flip(outputs, dims=[0])
            rev_out, _ = self.back_lstm(rev.unsqueeze(1))
            rev_out = rev_out.squeeze(1)
            rev_out = torch.flip(rev_out, dims=[0])
            outputs = torch.cat([outputs, rev_out], dim=-1)

        return outputs, (hx, cx)

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

    @staticmethod
    def from_pretrained(path: str) -> "QLSTMExtended":
        return torch.load(path)

class LSTMTaggerExtended(nn.Module):
    """Sequence tagging model that supports the extended QLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        bidirectional: bool = False,
        depth: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = (
            QLSTMExtended(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                bidirectional=bidirectional,
                depth=depth,
            )
            if n_qubits
            else nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        )
        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidirectional else 1), tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMExtended", "LSTMTaggerExtended"]
