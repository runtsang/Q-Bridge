from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Classical photonic‑style feature extractor
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters that emulate a single photonic layer."""
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
    """Create a sequential PyTorch model that mirrors the photonic architecture."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ----------------------------------------------------------------------
# Quantum circuit construction (incremental data‑uploading ansatz)
# ----------------------------------------------------------------------
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[qiskit.QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


# ----------------------------------------------------------------------
# Quantum execution wrapper
# ----------------------------------------------------------------------
class QuantumCircuitWrapper:
    """Thin wrapper around a Qiskit circuit that returns expectation values of Z on the first qubit."""

    def __init__(self, circuit: qiskit.QuantumCircuit, backend, shots: int = 1024):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.param_names = list(circuit.parameters)

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: val} for p, val in zip(self.param_names, params)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            exp = 0.0
            for state, count in counts.items():
                z = 1.0 if state[0] == "1" else -1.0
                exp += z * count
            return exp / self.shots

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


# ----------------------------------------------------------------------
# Differentiable hybrid layer
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Forward‑backward interface that sends data through a quantum circuit."""

    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float
    ):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(inputs.tolist())
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.numpy()):
            exp_r = ctx.circuit.run([val + shift[idx]])[0]
            exp_l = ctx.circuit.run([val - shift[idx]])[0]
            grads.append(exp_r - exp_l)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None


class QuantumHybridHead(nn.Module):
    """Dense head that forwards activations through a variational quantum circuit."""

    def __init__(
        self,
        circuit: qiskit.QuantumCircuit,
        backend,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ):
        super().__init__()
        self.wrapper = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.wrapper, self.shift)


# ----------------------------------------------------------------------
# Full hybrid fraud‑detection model
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """End‑to‑end hybrid model that combines a photonic feature extractor with a quantum head."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        num_qubits: int,
        depth: int,
        backend,
    ):
        super().__init__()
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        circuit, _, _, _ = build_classifier_circuit(num_qubits, depth)
        self.head = QuantumHybridHead(circuit, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.head(features)
