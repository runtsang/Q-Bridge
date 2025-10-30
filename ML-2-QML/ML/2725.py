import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np

# Qiskit imports for the quantum head
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import Parameter

@dataclass
class FraudLayerParameters:
    """Parameters describing a single dense layer in the classical backbone."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Build a single dense layer that mimics the photonic layer."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Return a sequential PyTorch model that mirrors the photonic circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuantumExpectationHead(nn.Module):
    """Two‑qubit quantum circuit that outputs a single expectation value."""
    def __init__(self, shots: int = 1024):
        super().__init__()
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        # Prepare a Bell‑like state
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.barrier()
        self.circuit.ry(theta, 0)
        self.circuit.ry(theta, 1)
        self.circuit.measure_all()
        self.theta = theta

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        thetas = inputs.detach().cpu().numpy()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: th} for th in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Compute expectation of Pauli‑Z on the first qubit
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k[0]) for k in counts.keys()])
            return np.sum((1 - 2 * states) * probs)
        if isinstance(result, list):
            exp = np.array([expectation(r) for r in result])
        else:
            exp = np.array([expectation(result)])
        return torch.tensor(exp, dtype=torch.float32)

class FraudDetectionHybridNet(nn.Module):
    """Hybrid fraud‑detection network with optional quantum head."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: List[FraudLayerParameters],
                 use_quantum_head: bool = False,
                 shots: int = 1024):
        super().__init__()
        self.backbone = build_fraud_detection_program(input_params, hidden_params)
        self.use_quantum_head = use_quantum_head
        if self.use_quantum_head:
            self.head = QuantumExpectationHead(shots=shots)
        else:
            self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudLayerParameters",
           "build_fraud_detection_program",
           "QuantumExpectationHead",
           "FraudDetectionHybridNet"]
