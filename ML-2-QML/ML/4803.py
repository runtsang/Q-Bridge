import torch
from torch import nn
import qiskit
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

@dataclass
class PhotonicLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _make_photonic_layer(params: PhotonicLayerParams, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias   = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class PhotonicBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return PhotonicBlock()

def build_photonic_seq(input_params: PhotonicLayerParams,
                       layers: Iterable[PhotonicLayerParams]) -> nn.Sequential:
    first = _make_photonic_layer(input_params, clip=False)
    rest  = [_make_photonic_layer(p, clip=True) for p in layers]
    return nn.Sequential(*[first, *rest, nn.Linear(2, 1)])

class ConvFeature(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold   = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        out = activations.mean(dim=(2, 3))
        return self.dropout(out)

class QLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int, backend=None, shots=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_wires)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_wires)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_wires)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_wires)

    def _quantum_expectation(self, angles: torch.Tensor) -> float:
        angles_np = angles.detach().cpu().numpy()
        n = len(angles_np)
        circ = qiskit.QuantumCircuit(n)
        for i, a in enumerate(angles_np):
            circ.rx(a, i)
        circ.measure_all()
        job = qiskit.execute(circ, backend=self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        total = 0
        for bitstring, c in counts.items():
            zeros = bitstring.count('0')
            ones = n - zeros
            exp += (zeros - ones) * c
            total += c
        return exp / total

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)
        f_raw = self.forget_lin(combined)
        i_raw = self.input_lin(combined)
        g_raw = self.update_lin(combined)
        o_raw = self.output_lin(combined)

        f = torch.sigmoid(torch.tensor(self._quantum_expectation(f_raw)))
        i = torch.sigmoid(torch.tensor(self._quantum_expectation(i_raw)))
        g = torch.tanh(torch.tensor(self._quantum_expectation(g_raw)))
        o = torch.sigmoid(torch.tensor(self._quantum_expectation(o_raw)))

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class FraudDetectionHybrid(nn.Module):
    def __init__(self,
                 photonic_params: Iterable[PhotonicLayerParams],
                 conv_kernel: int = 2,
                 n_qubits: int = 4,
                 hidden_dim: int = 16):
        super().__init__()
        self.photonic = build_photonic_seq(photonic_params[0], photonic_params[1:])
        self.conv = ConvFeature(kernel_size=conv_kernel)
        self.lstm = QLSTMCell(input_dim=1, hidden_dim=hidden_dim, n_wires=n_qubits)

    def forward(self, x: torch.Tensor):
        y = self.photonic(x)
        y_img = y.view(-1, 1, 1, 1).repeat(1, 1, self.conv.kernel_size, self.conv.kernel_size)
        conv_out = self.conv(y_img)
        hx = torch.zeros(x.size(0), self.lstm.hidden_dim, device=x.device)
        cx = torch.zeros(x.size(0), self.lstm.hidden_dim, device=x.device)
        hx, cx = self.lstm(conv_out, hx, cx)
        return hx

def build_fraud_detection_program(input_params: PhotonicLayerParams,
                                  layers: Iterable[PhotonicLayerParams]) -> nn.Sequential:
    return build_photonic_seq(input_params, layers)
