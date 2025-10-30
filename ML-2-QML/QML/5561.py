import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------
# Fraud‑detection inspired quantum layer parameters
# ----------------------------------------------------
@dataclass
class FraudLayerParameters:
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

# ----------------------------------------------------
# Quantum feature encoder
# ----------------------------------------------------
class QEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        self.encoder(qdev, x)

# ----------------------------------------------------
# Quantum transformer block
# ----------------------------------------------------
class QuantumTransformerBlock(tq.QuantumModule):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires: int,
                 dropout: float = 0.1):
        super().__init__()
        self.n_wires = n_wires
        self.attn = tq.QuantumModule()
        self.ffn = tq.QuantumModule()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Simple parameterised quantum layers for attention and feed‑forward
        self.attn_layer = tq.QuantumModule()
        self.attn_layer.register_buffer("n_wires", torch.tensor(n_wires))
        self.attn_layer.register_buffer("measure", torch.tensor(1.0))  # placeholder

        self.ffn_layer = tq.QuantumModule()
        self.ffn_layer.register_buffer("n_wires", torch.tensor(n_wires))
        self.ffn_layer.register_buffer("measure", torch.tensor(1.0))  # placeholder

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Flatten batch and seq for quantum processing
        batch, seq, dim = x.shape
        tokens = x.reshape(batch * seq, dim)
        # Quantum attention (placeholder: identity)
        attn_out = tokens
        attn_out = attn_out.reshape(batch, seq, dim)
        x = self.norm1(x + self.dropout(attn_out))
        # Quantum feed‑forward (placeholder: identity)
        ffn_out = tokens
        ffn_out = ffn_out.reshape(batch, seq, dim)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------
# Quantum fraud‑detection inspired head
# ----------------------------------------------------
def _quantum_layer_from_params(params: FraudLayerParameters, *, clip: bool) -> tq.QuantumModule:
    class Layer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.rx(qdev)
            self.ry(qdev)
            return self.measure(qdev)
    return Layer()

def build_fraud_detection_program_q(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    modules = [_quantum_layer_from_params(input_params, clip=False)]
    modules.extend(_quantum_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------
# Hybrid quantum regression model
# ----------------------------------------------------
class QModel(tq.QuantumModule):
    def __init__(self,
                 num_features: int,
                 n_wires: int = 8,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 64,
                 fraud_params: Sequence[FraudLayerParameters] = None):
        super().__init__()
        self.encoder = QEncoder(num_features)
        self.proj = nn.Linear(num_features, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, n_wires)
             for _ in range(num_blocks)]
        )
        if fraud_params:
            self.fraud_head = build_fraud_detection_program_q(fraud_params[0], fraud_params[1:])
        else:
            self.fraud_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=batch, device=x.device)
        self.encoder(qdev, x)
        features = tq.MeasureAll(tq.PauliZ)(qdev)
        x = self.proj(features)
        for block in self.transformer_blocks:
            x = block(x, qdev)
        x = x.mean(dim=1)
        return self.fraud_head(x)

# ----------------------------------------------------
# Dataset utilities (identical to classical for consistency)
# ----------------------------------------------------
def generate_superposition_data(num_features: int, samples: int):
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = [
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
    "FraudLayerParameters",
]
