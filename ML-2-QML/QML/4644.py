"""Hybrid quantum LSTM with fraud‑style preprocessing and quantum kernel mapping."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict


class FraudQuantumLayer(tq.QuantumModule):
    """Quantum layer that encodes classical parameters via RX gates and measures."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class FraudPreprocessorQuantum(nn.Module):
    """Sequential fraud‑style quantum preprocessing."""
    def __init__(self, n_wires: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FraudQuantumLayer(n_wires) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes two inputs via a fixed sequence of rotations."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        qdev.reset_states(x.shape[0])
        self.ansatz(qdev, x)
        self.ansatz(qdev, -y)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel computed via the ansatz."""
    def __init__(self, refs: Sequence[torch.Tensor], n_wires: int = 2) -> None:
        super().__init__()
        self.register_buffer("refs", torch.stack(list(refs)))
        self.n_wires = n_wires
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for ref in self.refs:
            self.ansatz(self.qdev, x, ref.unsqueeze(0))
            out.append(torch.abs(self.qdev.states.view(-1)[0]))
        return torch.stack(out, dim=-1)


class QuantumHybridQLSTM(nn.Module):
    """Quantum LSTM with fraud‑style preprocessing and quantum kernel mapping."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QuantumHybridTagger(nn.Module):
    """Tagger that chains fraud preprocessor, quantum kernel, and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        fraud_layers: int,
        kernel_refs: Sequence[torch.Tensor],
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.preprocessor = FraudPreprocessorQuantum(embedding_dim, fraud_layers)
        self.kernel_mapper = QuantumKernel(kernel_refs)
        kernel_dim = len(kernel_refs)
        self.lstm = QuantumHybridQLSTM(kernel_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        seq_len, batch, _ = embeds.shape
        flat = embeds.view(-1, embeds.shape[-1])
        pre = self.preprocessor(flat)
        kernel_vec = self.kernel_mapper(pre)
        kernel_vec = kernel_vec.view(seq_len, batch, -1)
        lstm_out, _ = self.lstm(kernel_vec)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumHybridQLSTM", "QuantumHybridTagger", "FraudPreprocessorQuantum", "QuantumKernel"]
