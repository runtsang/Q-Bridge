import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

def _build_photonic_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()

class ClassicalQLSTM(nn.Module):
    """Pure PyTorch LSTM cell used as a fallback when quantum resources are absent."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model combining photonic-inspired feature extraction
    with either a classical or a quantum-enhanced LSTM for sequence modeling."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: List[FraudLayerParameters],
                 hidden_dim: int = 32,
                 n_qubits: int = 0):
        super().__init__()
        self.extractor = nn.Sequential(
            _build_photonic_layer(input_params, clip=False),
            *[_build_photonic_layer(l, clip=True) for l in layers]
        )
        # Sequence modelling: try quantum LSTM if requested, otherwise fallback
        if n_qubits > 0:
            try:
                import importlib
                mod = importlib.import_module("FraudDetectionHybrid_qml")
                self.lstm = mod.QLSTM(2, hidden_dim, n_qubits)
            except Exception:
                self.lstm = ClassicalQLSTM(2, hidden_dim, n_qubits)
        else:
            self.lstm = ClassicalQLSTM(2, hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, 2)
        batch, seq_len, _ = x.shape
        # Feature extraction per timestep
        x_flat = x.reshape(batch * seq_len, 2)
        features = self.extractor(x_flat)
        features = features.reshape(batch, seq_len, -1)
        # Transpose to (seq_len, batch, 2) for LSTM compatibility
        features = features.permute(1, 0, 2)
        # Sequence modelling
        outputs, _ = self.lstm(features)
        logits = self.classifier(outputs[-1])
        return logits

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    hidden_dim: int = 32,
    n_qubits: int = 0,
) -> FraudDetectionHybrid:
    """Convenience factory that returns an instance of FraudDetectionHybrid."""
    return FraudDetectionHybrid(input_params, list(layers), hidden_dim, n_qubits)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_program"]
