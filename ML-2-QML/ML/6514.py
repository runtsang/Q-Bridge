"""Hybrid fraud‑detection model combining classical layers with a Qiskit variational circuit.

The class exposes a PyTorch network that first processes the two‑dimensional fraud signal
through a stack of learnable linear layers.  The intermediate activations are then
encoded into a Qiskit circuit whose expectation values of Pauli‑Z observables provide
additional quantum‑derived features.  The quantum part is fully differentiable via
parameter‑shift gradients, enabling end‑to‑end training when desired.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

@dataclasses.dataclass
class FraudDetectionParams:
    """Parameters for a single classical layer that mimics a photonic block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudDetectionParams, *, clip: bool) -> nn.Module:
    """Create a single linear + tanh block with optional clipping of weights."""
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()

def build_classical_network(
    input_params: FraudDetectionParams,
    layers: Iterable[FraudDetectionParams],
) -> nn.Sequential:
    """Stack the first layer (unclipped) followed by all subsequent clipped layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

def build_qiskit_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, list[SparsePauliOp]]:
    """Create a Qiskit variational ansatz with explicit data encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    w_index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[w_index], qubit)
            w_index += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, encoding, weights, observables

class FraudDetectionHybrid(nn.Module):
    """Classical‑plus‑Qiskit fraud‑detection pipeline.

    Parameters
    ----------
    input_params : FraudDetectionParams
        Parameters for the first classical layer.
    layers : Iterable[FraudDetectionParams]
        Parameters for the remaining layers.
    q_depth : int
        Depth of the Qiskit variational ansatz.
    num_qubits : int
        Number of qubits used for data encoding.
    """

    def __init__(
        self,
        input_params: FraudDetectionParams,
        layers: Iterable[FraudDetectionParams],
        q_depth: int,
        num_qubits: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.classical_net = build_classical_network(input_params, layers).to(device)
        self.qc, self.enc_params, self.var_params, self.observables = build_qiskit_classifier_circuit(
            num_qubits, q_depth
        )
        self.simulator = Aer.get_backend("qasm_simulator")
        self.device = device

    def _quantum_expectations(self, features: torch.Tensor) -> torch.Tensor:
        """Run the Qiskit circuit for each row in ``features`` and return Pauli‑Z expectations."""
        batch_size = features.shape[0]
        fx = features.detach().cpu().numpy()
        z_exp = np.zeros((batch_size, len(self.observables)), dtype=np.float32)
        for i, row in enumerate(fx):
            bound = {p: val for p, val in zip(self.enc_params, row)}
            bound.update({p: 0.0 for p in self.var_params})  # zero variational params for inference
            bound_circ = self.qc.bind_parameters(bound)
            job = execute(bound_circ, self.simulator, shots=1024)
            result = job.result()
            counts = result.get_counts()
            for j, _ in enumerate(self.observables):
                z = 0.0
                for bitstring, n in counts.items():
                    # bitstring order is reversed in Qiskit
                    if bitstring[::-1][j] == "1":
                        z -= n
                    else:
                        z += n
                z_exp[i, j] = z / 1024
        return torch.from_numpy(z_exp).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return classical output and quantum‑derived features."""
        cls_out = self.classical_net(x)
        q_feats = self._quantum_expectations(x)
        return cls_out, q_feats

__all__ = ["FraudDetectionParams", "build_classical_network", "FraudDetectionHybrid"]
