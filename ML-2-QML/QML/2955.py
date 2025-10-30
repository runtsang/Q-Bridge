"""Quantum components for FraudDetectionHybridNet.

Provides a Strawberry‑Fields photonic program and a Qiskit variational circuit
that can be plugged into the classical network via a differentiable
HybridLayer.  The module is fully importable and does not depend on the
classical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Photonic layer parameters and program
# ------------------------------------------------------------------
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> "sf.Program":
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

# ------------------------------------------------------------------
# 2. Qiskit variational circuit
# ------------------------------------------------------------------
class QiskitFraudCircuit:
    """
    Two‑qubit parameterised circuit that emulates the photonic layer.
    Parameters are encoded as rotation angles on each qubit followed by a CZ gate.
    """
    def __init__(self, backend=None, shots: int = 1024):
        import qiskit
        from qiskit import Aer

        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        theta = qiskit.circuit.Parameter("theta")
        phi = qiskit.circuit.Parameter("phi")
        # Simple ansatz: RX(theta) on qubit 0, RZ(phi) on qubit 1, CZ
        self.circuit.rx(theta, 0)
        self.circuit.rz(phi, 1)
        self.circuit.cz(0, 1)
        self.circuit.measure_all()
        self.theta = theta
        self.phi = phi

    def run(self, angles: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Execute the circuit for a batch of (theta, phi) pairs.
        Returns a torch tensor of expectation values of Pauli Z on qubit 0.
        """
        import numpy as np
        from qiskit import assemble, transpile

        param_dicts = [{self.theta: a[0], self.phi: a[1]} for a in angles]
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, parameter_binds=param_dicts, shots=self.shots)
        job = self.backend.run(qobj)
        results = job.result().get_counts()

        def expectation(counts):
            # Convert counts dict to expectation of Z on qubit 0
            total = sum(counts.values())
            z_plus = sum(v for k, v in counts.items() if k[0] == "0")
            z_minus = total - z_plus
            return (z_plus - z_minus) / total

        if isinstance(results, list):
            return torch.tensor([expectation(r) for r in results], dtype=torch.float32)
        else:
            return torch.tensor([expectation(results)], dtype=torch.float32)

# ------------------------------------------------------------------
# 3. Differentiable hybrid layer
# ------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards inputs through a quantum circuit
    and returns the expectation value.  The backward pass uses a simple
    finite‑difference approximation.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QiskitFraudCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert inputs to list of (theta, phi) pairs
        angles = [(float(a[0]), float(a[1] + shift)) for a in inputs.tolist()]
        expectations = circuit.run(angles)
        ctx.save_for_backward(inputs)
        return expectations

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        # Finite difference approximation
        eps = 1e-3
        angles_plus = [(float(a[0]), float(a[1] + shift + eps)) for a in inputs.tolist()]
        angles_minus = [(float(a[0]), float(a[1] + shift - eps)) for a in inputs.tolist()]
        exp_plus = ctx.circuit.run(angles_plus)
        exp_minus = ctx.circuit.run(angles_minus)
        grad = (exp_plus - exp_minus) / (2 * eps)
        return grad * grad_output, None, None

class HybridLayer(nn.Module):
    """
    Integrates a QiskitFraudCircuit into a PyTorch module.
    """
    def __init__(self, circuit: QiskitFraudCircuit, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QiskitFraudCircuit",
    "HybridLayer",
]
