import torch
from torch import nn
from typing import Iterable, Tuple

# Classical photonic‑style fraud detection building blocks
class FraudLayerParameters:
    def __init__(self, bs_theta: float, bs_phi: float, phases: Tuple[float, float],
                 squeeze_r: Tuple[float, float], squeeze_phi: Tuple[float, float],
                 displacement_r: Tuple[float, float], displacement_phi: Tuple[float, float],
                 kerr: Tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip_val(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Classical convolution‑inspired network (derived from QCNN)
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Classical LSTM (drop‑in replacement)
class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement for the quantum LSTM using plain linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# Classical RBF kernel (placeholder)
class ClassicalKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# Hybrid fraud‑detection model
class FraudDetectionHybrid(nn.Module):
    """
    Combines classical photonic‑style layers, a convolution‑inspired network,
    an optional quantum LSTM, and a quantum kernel into a single classifier.
    """
    def __init__(
        self,
        input_dim: int = 2,
        seq_dim: int = 1,
        hidden_dim: int = 32,
        n_qubits: int = 4,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Classical photonic‑style fraud net
        default_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, 0.2),
            squeeze_r=(0.2, 0.3),
            squeeze_phi=(0.4, 0.5),
            displacement_r=(0.6, 0.7),
            displacement_phi=(0.8, 0.9),
            kerr=(0.1, 0.2),
        )
        self.fraud_net = build_fraud_detection_program(default_params, [])

        # Classical convolution‑inspired network
        self.cnn = QCNNModel()

        # LSTM: quantum if qubits supplied, else classical
        if n_qubits > 0:
            from.qml_module import QLSTMQuantumLayer  # lazy import
            self.lstm = QLSTMQuantumLayer(n_wires=n_qubits)
            lstm_out_dim = n_qubits
        else:
            self.lstm = ClassicalQLSTM(seq_dim, hidden_dim, n_qubits)
            lstm_out_dim = hidden_dim

        # Kernel: quantum if qubits supplied, else classical
        if n_qubits > 0:
            from.qml_module import QuantumKernel
            self.kernel = QuantumKernel()
            kernel_out_dim = 1
        else:
            self.kernel = ClassicalKernel(gamma)
            kernel_out_dim = 1

        # Final classifier
        self.classifier = nn.Linear(lstm_out_dim + 3, 1)

    def forward(
        self,
        features: torch.Tensor,
        seq: torch.Tensor,
        feature_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param features: Tensor of shape (batch, 2) fed to fraud_net and cnn.
        :param seq: Tensor of shape (batch, seq_len, seq_dim) for LSTM.
        :param feature_vec: Tensor of shape (batch, feature_dim) for kernel.
        """
        # Fraud net output
        fraud_out = self.fraud_net(features)  # (batch,1)

        # Convolution‑inspired output
        cnn_out = self.cnn(features)  # (batch,1)

        # LSTM output
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(seq)
            lstm_last = lstm_out[-1]  # (batch, hidden_dim)
        else:  # quantum LSTM
            seq_t = seq.transpose(0,1)  # (seq_len, batch, seq_dim)
            lstm_out, _ = self.lstm(seq_t)
            lstm_last = lstm_out[-1]  # (batch, n_qubits)

        # Kernel output
        kernel_out = self.kernel(feature_vec, feature_vec)  # (batch,1)

        # Concatenate and classify
        combined = torch.cat([fraud_out, cnn_out, lstm_last, kernel_out], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QCNNModel",
    "ClassicalQLSTM",
    "ClassicalKernel",
    "FraudDetectionHybrid",
]
