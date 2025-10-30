import numpy as np
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Optional

class HybridLayer(nn.Module):
    """
    Quantum‑enhanced hybrid layer that replaces the projection step of the
    attention head with a variational quantum circuit built with TorchQuantum.
    The interface mirrors the classical seed: a `run` method that accepts
    parameters and returns a NumPy array.  The quantum part uses a random
    layer followed by parametrized RX, RY, RZ, CRX gates and measurement
    of Pauli‑Z.
    """

    def __init__(self,
                 mode: str = "quantum",
                 n_features: int = 1,
                 embed_dim: int = 4,
                 n_heads: int = 1,
                 n_qubits: int = 4,
                 conv_channels: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits

        if mode == "quantum":
            self.q_layer = self._build_q_layer()
            self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        elif mode == "conv":
            if conv_channels is None:
                conv_channels = 8
            self.features = nn.Sequential(
                nn.Conv2d(1, conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")

        self.final = nn.Tanh()

    def _build_q_layer(self):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)
                self.rz = tq.RZ(has_params=True, trainable=True)
                self.crx = tq.CRX(has_params=True, trainable=True)
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
                self.random_layer(qdev)
                self.rx(qdev, wires=0)
                self.ry(qdev, wires=1)
                self.rz(qdev, wires=2)
                self.crx(qdev, wires=[0, 1])
                return self.measure(qdev)

        return QLayer(self.n_qubits)

    def forward(self, x: torch.Tensor, thetas: Iterable[float] = None) -> torch.Tensor:
        if self.mode == "quantum":
            # Assume x shape (batch, seq, embed)
            batch, seq, embed = x.shape
            outputs = []
            for token in x.unbind(dim=1):
                token = token.unsqueeze(0)  # shape (1, embed)
                qdev = self.q_device.copy(bsz=token.shape[0], device=token.device)
                out = self.q_layer(token, qdev)
                outputs.append(out)
            out = torch.stack(outputs, dim=1).mean(dim=1).unsqueeze(-1)
        elif self.mode == "conv":
            features = self.features(x)
            flattened = features.view(features.shape[0], -1)
            out = self.fc(flattened)
        else:
            raise RuntimeError("unreachable")

        return self.final(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        if self.mode == "conv":
            dummy = torch.zeros(1, 1, 28, 28)
        else:
            dummy = torch.zeros(1, 1, self.embed_dim)
        out = self.forward(dummy, thetas)
        return out.detach().cpu().numpy()

__all__ = ["HybridLayer"]
