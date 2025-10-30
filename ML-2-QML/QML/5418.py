"""QuantumNATEnhanced – quantum counterpart using torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf

__all__ = ["QuantumNATEnhanced"]


class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum implementation that mirrors the classical `QuantumNATEnhanced` API.
    Each block is replaced by a parameterised quantum circuit that operates on
    a fixed number of wires:

    * 2‑D image encoder → `GeneralEncoder` (4‑wire RyZXY) + random layer.
    * Regression head → simple quantum layer followed by a linear read‑out.
    * LSTM tagger → quantum gates inside a `QLayer` that mimics the classical LSTM gates.
    * Sampler → a 2‑qubit parameterised circuit with a state‑vector sampler.

    The forward methods accept the same tensor shapes as the classical version
    and return quantum expectations or sampled probabilities.
    """

    class _ImageEncoder(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))

        def forward(self, qdev: tq.QuantumDevice, features: torch.Tensor) -> None:
            # `features` shape: (bsz, 16) – flatten from pooled image
            self.random(qdev)
            self.encoder(qdev, features)

    class _RegressionLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random(qdev)
            self.rx(qdev)
            self.ry(qdev)
            return self.measure(qdev)

    class _QLayer(tq.QuantumModule):
        """Quantum gate block that is used inside the quantum LSTM."""
        def __init__(self, n_wires: int = 4):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    class _LSTMCell(tq.QuantumModule):
        """Quantum‑based LSTM cell that replaces classical gates."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.forget = QuantumNATEnhanced._QLayer(n_wires)
            self.input = QuantumNATEnhanced._QLayer(n_wires)
            self.update = QuantumNATEnhanced._QLayer(n_wires)
            self.output = QuantumNATEnhanced._QLayer(n_wires)

            self.linear_forget = nn.Linear(4 + 4, n_wires)
            self.linear_input = nn.Linear(4 + 4, n_wires)
            self.linear_update = nn.Linear(4 + 4, n_wires)
            self.linear_output = nn.Linear(4 + 4, n_wires)

            self.head = nn.Linear(n_wires, 10)  # tagset size

        def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            return hx, cx

    class _Sampler(tq.QuantumModule):
        """2‑qubit parameterised circuit that outputs a probability distribution."""
        def __init__(self):
            super().__init__()
            self.n_wires = 2
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                ]
            )
            self.cnot = tq.CNOT()
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice, inp: torch.Tensor) -> torch.Tensor:
            self.encoder(qdev, inp)
            self.cnot(qdev)
            self.rx(qdev)
            self.ry(qdev)
            return self.measure(qdev)

    def __init__(self, n_lstm_layers: int = 1, lstm_hidden: int = 32) -> None:
        super().__init__()
        self.n_wires = 4
        self.image_encoder = QuantumNATEnhanced._ImageEncoder(self.n_wires)
        self.regression_layer = QuantumNATEnhanced._RegressionLayer(self.n_wires)
        self.lstm_cell = QuantumNATEnhanced._LSTMCell(self.n_wires)
        self.sampler = QuantumNATEnhanced._Sampler()

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------
    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (bsz, 1, 28, 28) grayscale image batch
        :return: (bsz, 4) quantum expectation vector
        """
        bsz = x.shape[0]
        # Average‑pool to 16‑dim vector (same as classical avg_pool2d(6))
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.image_encoder(qdev, pooled)
        return self.regression_layer(qdev).squeeze(-1)

    # ------------------------------------------------------------------
    # Regression forward
    # ------------------------------------------------------------------
    def forward_regression(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        :param image_embedding: (bsz, 4) from `forward_image`
        :return: (bsz,) scalar predictions
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=image_embedding.shape[0], device=image_embedding.device)
        self.regression_layer(qdev)
        return self.regression_layer.measure(qdev).squeeze(-1)

    # ------------------------------------------------------------------
    # Sequence tagging forward
    # ------------------------------------------------------------------
    def forward_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        :param seq: (bsz, seq_len, 4) sequence of embeddings
        :return: (bsz, seq_len, tagset_size)
        """
        bsz, seq_len, _ = seq.shape
        hx = torch.zeros(bsz, self.n_wires, device=seq.device)
        cx = torch.zeros(bsz, self.n_wires, device=seq.device)
        outputs = []
        for t in range(seq_len):
            hx, cx = self.lstm_cell(seq[:, t, :], hx, cx)
            outputs.append(hx.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)
        logits = self.lstm_cell.head(out_seq)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Sampler forward
    # ------------------------------------------------------------------
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (bsz, 2) raw inputs for the 2‑qubit circuit
        :return: (bsz, 2) probability distribution
        """
        bsz = inputs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.sampler.n_wires, bsz=bsz, device=inputs.device)
        probs = self.sampler(qdev, inputs)
        return probs
