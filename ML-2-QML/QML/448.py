"""Quantum‑enhanced LSTM with depth‑controlled circuits and optional noise.

The module mirrors the classical API but replaces each gate with a
parameterised quantum circuit.  The depth of the circuit can be tuned,
and a depolarising noise model can be injected to emulate real hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Parameterized quantum layer with optional depth and noise."""

    def __init__(
        self,
        n_wires: int,
        depth: int = 0,
        noise_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.noise_rate = noise_rate

        # Encode each input dimension as an RX gate on its own wire.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        if noise_rate > 0.0:
            self.noise = tq.NoiseModel(
                depolarizing_noise_rate=noise_rate,
                n_qubits=n_wires,
            )
        else:
            self.noise = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        if self.noise is not None:
            qdev.noise_model = self.noise
        # Input encoding
        self.encoder(qdev, x)
        # Apply trainable RX gates
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Depth‑controlled CNOT layers
        for _ in range(self.depth):
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            # Optional additional RX gates after each depth
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
        return self.measure(qdev)


class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a quantum circuit."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 0,
        noise_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits, depth=depth, noise_rate=noise_rate)
        self.input = QLayer(n_qubits, depth=depth, noise_rate=noise_rate)
        self.update = QLayer(n_qubits, depth=depth, noise_rate=noise_rate)
        self.output = QLayer(n_qubits, depth=depth, noise_rate=noise_rate)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 0,
        noise_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                depth=depth,
                noise_rate=noise_rate,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
