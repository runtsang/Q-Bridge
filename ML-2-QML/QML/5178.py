"""Quantum hybrid model integrating quanvolution, QCNN, binary classification, and Quantum‑NAT concepts."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf


class QuantumExpectationCircuit(tq.QuantumModule):
    """Simple one‑qubit circuit measuring the expectation of Z after an Ry rotation."""
    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "ry", "wires": [0]}]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        return self.measure(qdev)


class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumExpectationCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit(inputs)
        ctx.save_for_backward(inputs, expectation)
        return expectation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, expectation = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs:
            right = ctx.circuit(val + shift).item()
            left = ctx.circuit(val - shift).item()
            grad_inputs.append(right - left)
        grad = torch.tensor(grad_inputs).float()
        return grad * grad_output, None, None


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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


class QCNNBlock(tq.QuantumModule):
    """Quantum QCNN‑style block performing convolution and pooling on 4 qubits."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder mapping each input feature to a ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Convolutional layer using a random unitary
        self.conv_layer = tq.RandomLayer(n_ops=8, wires=[0,1,2,3])
        # Pooling layer using another random unitary
        self.pool_layer = tq.RandomLayer(n_ops=4, wires=[0,1,2,3])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.conv_layer(qdev)
        self.pool_layer(qdev)
        measurement = self.measure(qdev)
        return measurement


class QuanvolutionHybrid(nn.Module):
    """Hybrid model combining a quantum quanvolution filter, a QCNN‑style block,
    and a quantum expectation head for binary classification."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, shift: float = 0.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Reduce high‑dimensional quantum features to 4 for QCNN
        self.reduce = nn.Linear(4 * 14 * 14, 4)
        self.qcnn = QCNNBlock()
        self.linear = nn.Linear(4, 1)
        self.circuit = QuantumExpectationCircuit()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.qfilter(x)          # (bsz, 4*14*14)
        qfeat_reduced = self.reduce(qfeat)  # (bsz, 4)
        qcnn_out = self.qcnn(qfeat_reduced)  # (bsz, 4)
        logits = self.linear(qcnn_out)  # (bsz, 1)
        probs = HybridFunction.apply(logits.squeeze(), self.circuit, self.shift)  # (bsz,)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["QuanvolutionHybrid"]
