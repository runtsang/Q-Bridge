"""Hybrid classical-quantum model combining CNN, quantum fully-connected layer, fraud detection layers, and estimator head.

The model can operate in fully classical mode or hybrid mode with a TorchQuantum submodule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

# Optional import of TorchQuantum
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    _HAS_TQ = True
except Exception:  # pragma: no cover
    _HAS_TQ = False

class HybridQuantumNAT(nn.Module):
    """
    Hybrid model inspired by Quantum-NAT, FraudDetection, and EstimatorQNN.
    Parameters
    ----------
    use_quantum : bool
        If True and TorchQuantum is available, uses a quantum fully-connected
        layer; otherwise falls back to a classical linear layer.
    fraud_layer_count : int
        Number of fraud-detection style layers to stack after the FC layer.
    estimator_head : bool
        If True, appends a small regression head (EstimatorQNN style).
    """
    def __init__(self,
                 use_quantum: bool = True,
                 fraud_layer_count: int = 2,
                 estimator_head: bool = True,
                 device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        # Feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_dim = 16 * 7 * 7  # assuming input 28x28

        # Quantum or classical fully connected block
        self.use_quantum = use_quantum and _HAS_TQ
        if self.use_quantum:
            self.qfc = _QuantumFullyConnectedLayer(n_wires=4)
        else:
            self.qfc = nn.Linear(self.feature_dim, 4)

        self.norm = nn.BatchNorm1d(4)

        # Fraud detection style layers
        self.fraud_layers = nn.ModuleList()
        for _ in range(fraud_layer_count):
            self.fraud_layers.append(_FraudLayer(in_features=4))

        # Estimator head
        if estimator_head:
            self.estimator = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        else:
            self.estimator = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)

        if self.use_quantum:
            # prepare quantum device
            qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=self.device, record_op=True)
            # encode features into quantum state
            self.qfc.encode(qdev, flattened)
            out = self.qfc(qdev)
        else:
            out = self.qfc(flattened)

        out = self.norm(out)

        # Fraud detection layers
        for layer in self.fraud_layers:
            out = layer(out)

        if self.estimator is not None:
            out = self.estimator(out)

        return out

class _QuantumFullyConnectedLayer(tq.QuantumModule):
    """Quantum fully-connected layer used when `use_quantum=True`."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def encode(self, qdev: tq.QuantumDevice, features: torch.Tensor) -> None:
        """Map a feature vector onto the first qubits via Ry rotations."""
        # simple encoding: use the first 4 feature components
        for i in range(self.n_wires):
            tqf.ry(qdev, features[:, i], wires=[i])

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])
        out = self.measure(qdev)
        return self.norm(out)

class _FraudLayer(nn.Module):
    """Replicates the FraudDetection linear+activation+scale+shift block."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        weight = torch.randn(2, in_features)
        bias = torch.randn(2)
        self.linear = nn.Linear(in_features, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.randn(2))
        self.shift = nn.Parameter(torch.randn(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear(inputs))
        outputs = outputs * self.scale + self.shift
        return outputs

__all__ = ["HybridQuantumNAT"]
