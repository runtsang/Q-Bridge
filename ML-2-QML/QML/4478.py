from __future__ import annotations
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumLSTM(nn.Module):
    """
    LSTM where each gate is a small variational quantum circuit.
    The implementation follows the structure of the classical LSTM
    but replaces the linear projections with quantum encoders.
    """

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encoder that maps classical input into rotation angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "rz", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            # Trainable rotation layer
            self.trainable = tq.RandomLayer(
                n_ops=10, wires=list(range(n_wires))
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.trainable(qdev)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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

class QuantumHybridHead(nn.Module):
    """
    A small quantum circuit that acts as a classification head.
    It encodes the input vector into rotation angles, applies a
    trainable variational layer, and returns a single expectation
    value that is interpreted as a probability.
    """

    def __init__(self, input_dim: int, n_qubits: int = 4, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.trainable = tq.RandomLayer(
            n_ops=20, wires=list(range(n_qubits))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # flatten input to match encoder expectations
        batch = inputs.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=inputs.device, record_op=True)
        self.encoder(qdev, inputs)
        out = self.trainable(qdev)
        return self.measure(qdev).unsqueeze(-1)

class FraudQuantumLayer(tq.QuantumModule):
    """
    Quantum analogue of the photonic fraud layer.
    It encodes the two input features into a small circuit and
    outputs a 2‑dimensional vector that can be fed into a
    subsequent quantum or classical layer.
    """

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.trainable = tq.RandomLayer(
            n_ops=5, wires=list(range(n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.trainable(qdev)
        return self.measure(qdev)

class HybridQLSTMTagger(nn.Module):
    """
    Wrapper that stitches together the quantum LSTM, an optional
    fraud‑style preprocessing layer, and a quantum classification head.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 8,
        use_fraud_layer: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.fraud_layer = FraudQuantumLayer(n_wires=2) if use_fraud_layer else None
        self.head = QuantumHybridHead(hidden_dim, n_qubits=hidden_dim)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if self.fraud_layer is not None:
            # apply fraud quantum layer to each hidden state
            batch = lstm_out.size(0)
            qdev = tq.QuantumDevice(n_wires=2, bsz=batch, device=lstm_out.device, record_op=True)
            self.fraud_layer(qdev)
            lstm_out = self.fraud_layer.measure(qdev).view(batch, -1)
        logits = self.head(lstm_out)
        return logits

__all__ = [
    "QuantumLSTM",
    "QuantumHybridHead",
    "FraudQuantumLayer",
    "HybridQLSTMTagger",
]
