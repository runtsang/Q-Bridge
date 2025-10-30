"""
FraudDetectorHybrid: classical photonic backbone + quantum self‑attention + quantum‑enhanced LSTM.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Quantum modules live in qml_module (imported lazily to keep ml code independent)
try:
    from qml_module import QuantumSelfAttention, QLSTM
except ImportError:  # pragma: no cover
    # Dummy placeholders for type‑checking; real implementations are in qml_module
    class QuantumSelfAttention(nn.Module):
        def __init__(self, n_qubits: int = 4):
            super().__init__()
            self.n_qubits = n_qubits
        def run(self, rotation_params, entangle_params, inputs):
            return inputs  # identity

    class QLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_qubits):
            super().__init__()
            self.hidden_dim = hidden_dim
        def forward(self, inputs, states=None):
            return inputs, (torch.zeros_like(inputs), torch.zeros_like(inputs))

# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhotonicLayerParams:
    """Parameters that encode a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float] = (0.0, 0.2)

class PhotonicLayer(nn.Module):
    """Differentiable layer that mimics the optics of a photonic circuit."""
    def __init__(self, params: PhotonicLayerParams, clip: bool = False):
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

def build_fraud_detection_program(
    input_params: PhotonicLayerParams,
    layers: Iterable[PhotonicLayerParams]
) -> nn.Sequential:
    """Create a sequential model consisting of the photonic backbone."""
    modules: List[nn.Module] = [PhotonicLayer(input_params, clip=False)]
    modules.extend(PhotonicLayer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))  # Final classification head
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
class FraudDetectorHybrid(nn.Module):
    """
    End‑to‑end fraud detection model that chains a photonic backbone,
    a quantum self‑attention block, and a quantum‑enhanced LSTM tagger.
    """
    def __init__(
        self,
        photonic_params: List[PhotonicLayerParams],
        attention_n_qubits: int = 4,
        lstm_hidden_dim: int = 32,
        lstm_n_qubits: int = 4,
    ):
        super().__init__()
        self.backbone = build_fraud_detection_program(photonic_params[0], photonic_params[1:])
        self.attention = QuantumSelfAttention(attention_n_qubits)
        # LSTM tagger: project backbone output to hidden dim, then QLSTM, then output layer
        self.lstm_tagger = nn.Sequential(
            nn.Linear(1, lstm_hidden_dim),
            QLSTM(lstm_hidden_dim, lstm_hidden_dim, lstm_n_qubits),
            nn.Linear(lstm_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, 2) – raw transaction features.
        Returns:
            Tensor of shape (seq_len, batch, 1) – fraud probability scores.
        """
        seq_len, batch, _ = x.shape
        x_flat = x.view(seq_len * batch, -1)
        backbone_out = self.backbone(x_flat).view(seq_len, batch, -1)

        # Quantum self‑attention – for demo use random parameters
        rotation_params = torch.randn(self.attention.n_qubits * 3, dtype=torch.float32)
        entangle_params = torch.randn(self.attention.n_qubits - 1, dtype=torch.float32)
        attn_weights = self.attention.run(rotation_params, entangle_params, backbone_out)
        if attn_weights.shape!= backbone_out.shape:
            attn_weights = attn_weights.view(seq_len, batch, -1)
        attended = backbone_out * attn_weights

        lstm_out, _ = self.lstm_tagger(attended)
        return lstm_out
