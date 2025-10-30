"""Quantum hybrid model that mirrors the classical quanvolution with a quantum kernel
and a regression head inspired by EstimatorQNN."""

import torch
import torchquantum as tq
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a variational quantum kernel to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8 * n_layers, wires=list(range(n_wires)))
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
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)  # shape (bsz, n_wires*14*14)

class EstimatorQNN(tq.QuantumModule):
    """Simple variational estimator that maps quantum features to a scalar."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.param_layer = tq.RandomLayer(n_ops=4, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, n_features)
        batch = x.shape[0]
        self.qdev = tq.QuantumDevice(self.n_qubits, bsz=batch, device=x.device)
        x = x.view(batch, -1, self.n_qubits)
        outputs = []
        for i in range(x.shape[1]):
            self.param_layer(self.qdev, x[:, i, :])
            meas = self.measure(self.qdev)
            outputs.append(meas)
        return torch.stack(outputs, dim=1).sum(dim=1).unsqueeze(-1)

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum counterpart of the classical QuanvolutionHybrid."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2, n_qubits: int = 4):
        super().__init__()
        self.filter = QuanvolutionFilter(n_wires, n_layers)
        self.estimator = EstimatorQNN(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.estimator(features)
        return logits

__all__ = ["QuanvolutionHybrid"]
