from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


@dataclass
class FraudLayerParameters:
    """Parameters that describe a photonic fraud‑layer, reused as a quantum feature extractor."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumPhotonicFeatureExtractor(tq.QuantumModule):
    """Quantum photonic circuit that mirrors the classical fraud‑detection layers."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers = list(layers)

    def _apply_layer(self, qdev: tq.QuantumDevice, params: FraudLayerParameters, *, clip: bool) -> None:
        # Beam splitter
        tq.BSgate(params.bs_theta, params.bs_phi)(qdev, wires=[0, 1])
        # Phase rotations
        for i, phase in enumerate(params.phases):
            tq.Rgate(phase)(qdev, wires=[i])
        # Squeezing
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            tq.Sgate(r if not clip else _clip(r, 5), phi)(qdev, wires=[i])
        # Second beam splitter
        tq.BSgate(params.bs_theta, params.bs_phi)(qdev, wires=[0, 1])
        # Phase rotations again
        for i, phase in enumerate(params.phases):
            tq.Rgate(phase)(qdev, wires=[i])
        # Displacement
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            tq.Dgate(r if not clip else _clip(r, 5), phi)(qdev, wires=[i])
        # Kerr
        for i, k in enumerate(params.kerr):
            tq.Kgate(k if not clip else _clip(k, 1))(qdev, wires=[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the 2‑dimensional input as X‑rotations and run all photonic layers."""
        batch_size = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=2, bsz=batch_size, device=x.device)
        # Encode input as X rotations
        tq.Rgate(x[:, 0])(qdev, wires=[0])
        tq.Rgate(x[:, 1])(qdev, wires=[1])
        # Apply input layer
        self._apply_layer(qdev, self.input_params, clip=False)
        # Apply subsequent layers
        for layer in self.layers:
            self._apply_layer(qdev, layer, clip=True)
        # Measure all qubits in Z basis
        return tq.MeasureAll(tq.PauliZ)(qdev)


class QLSTM(nn.Module):
    """Quantum‑style LSTM cell with gates implemented by small quantum modules."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridQLSTMTagger(nn.Module):
    """Sequence tagging model that uses a quantum photonic feature extractor followed by a quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.preprocessor = QuantumPhotonicFeatureExtractor(input_params, layer_params)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        preprocessed = self.preprocessor(embeds)
        lstm_out, _ = self.lstm(preprocessed.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = [
    "FraudLayerParameters",
    "QuantumPhotonicFeatureExtractor",
    "QLSTM",
    "HybridQLSTMTagger",
]
