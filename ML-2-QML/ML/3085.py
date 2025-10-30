from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

@dataclass
class FraudLayerParameters:
    """Photonic‑inspired layer parameters."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # new optional fields
    dropout: float = 0.0
    batch_norm: bool = False

class QuantumConvFilter(nn.Module):
    """Quantum convolution implemented with a Qiskit circuit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 256):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        n = self.kernel_size ** 2
        qc = qiskit.QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply quantum convolution to a 2‑D input tensor."""
        batch, _, h, w = data.shape
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1
        out = torch.zeros((batch, out_h, out_w), device=data.device, dtype=torch.float32)

        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    patch = data[b, 0, i:i+self.kernel_size, j:j+self.kernel_size].cpu().numpy()
                    val = self._run_patch(patch)
                    out[b, i, j] = val
        return out.unsqueeze(1)  # add channel dimension

    def _run_patch(self, patch: np.ndarray) -> float:
        flat = patch.flatten()
        param_bind = {f"theta{i}": np.pi if v > self.threshold else 0 for i, v in enumerate(flat)}
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts()
        total = self.shots * self._circuit.num_qubits
        ones = sum(k.count('1') * v for k, v in counts.items())
        return ones / total

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model: quantum convolution + photonic‑inspired FC layers."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        use_quantum_conv: bool = True,
    ) -> None:
        super().__init__()
        modules = []
        if use_quantum_conv:
            modules.append(QuantumConvFilter(kernel_size=2, threshold=0.5, shots=128))
        modules.append(nn.Flatten())
        modules.append(_layer_from_params(input_params, clip=False))
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: list[FraudLayerParameters]) -> FraudDetectionHybrid:
    """Compatibility wrapper matching the original anchor."""
    return FraudDetectionHybrid(input_params, layers)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_program"]
