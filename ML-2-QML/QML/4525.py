"""Hybrid quantum neural architecture combining quantum convolution (quanvolution), quantum LSTM, and quantum kernel classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """Quantum version of HybridNATModel.

    The architecture mirrors the classical version but replaces all key components with quantum counterparts:
    - QuanvolutionFilter for feature extraction
    - QLSTM for sequence encoding
    - RBF kernel for classification
    """

    class QuanvolutionFilter(tq.QuantumModule):
        """Quantum 2x2 patch encoder using a random two‑qubit layer."""

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            x = x.view(bsz, 28, 28)
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    data = torch.stack(
                        [
                            x[:, r, c],
                            x[:, r, c + 1],
                            x[:, r + 1, c],
                            x[:, r + 1, c + 1],
                        ],
                        dim=1,
                    )
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    measurement = self.measure(qdev)
                    patches.append(measurement.view(bsz, 4))
            return torch.cat(patches, dim=1)

    class QLSTM(tq.QuantumModule):
        """Quantum LSTM cell with gates implemented by small quantum circuits."""

        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
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
                qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires):
                    if wire == self.n_wires - 1:
                        tqf.cnot(qdev, wires=[wire, 0])
                    else:
                        tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)

        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

        def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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

        def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )

    def __init__(
        self,
        n_qubits: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 10,
        kernel_gamma: float = 1.0,
        n_support: int = 32,
    ) -> None:
        super().__init__()
        self.filter = self.QuanvolutionFilter()
        self.lstm = self.QLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.kernel_gamma = kernel_gamma
        self.support_vectors = nn.Parameter(torch.randn(n_support, hidden_dim))
        self.classifier_weights = nn.Parameter(torch.randn(n_support, num_classes))

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.kernel_gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # Quanvolution
        features = self.filter(x)  # (bsz, 196*4)
        seq = features.view(bsz, 196, 4).permute(1, 0, 2)  # (196, bsz, 4)
        # Quantum LSTM
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[-1]  # (bsz, hidden_dim)
        # Kernel similarity
        kernel_mat = self._rbf_kernel(last_hidden, self.support_vectors)
        logits = kernel_mat @ self.classifier_weights
        return torch.log_softmax(logits, dim=1)

__all__ = ["HybridNATModel"]
