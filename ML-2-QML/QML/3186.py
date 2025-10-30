"""Unified hybrid layer combining classical fully‑connected and photonic fraud‑detection motifs with a quantum circuit."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable

import qiskit
import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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

class _QuantumCircuit:
    """Simple parameterized quantum circuit for a fully‑connected layer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class UnifiedHybridLayer(nn.Module):
    """
    Hybrid layer that combines a classical fraud‑detection submodule and a
    parameterized quantum circuit.  The quantum part is evaluated on a
    qiskit simulator; the classical part mirrors the structure of the
    fraud‑detection network.
    """

    def __init__(
        self,
        classical_params: FraudLayerParameters,
        n_qubits: int = 1,
        shots: int = 100,
        clip: bool = False,
    ) -> None:
        super().__init__()
        self.classical_submodule = _layer_from_params(classical_params, clip=clip)
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        self.quantum_circuit = _QuantumCircuit(n_qubits, simulator, shots)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the classical output and the quantum expectation value
        then concatenate them into a single numpy array.
        """
        dummy_input = torch.zeros(2, dtype=torch.float32)
        classical_output = self.classical_submodule(dummy_input).detach().numpy()
        quantum_output = self.quantum_circuit.run(thetas)
        return np.concatenate([classical_output, quantum_output], axis=0)

__all__ = ["UnifiedHybridLayer", "FraudLayerParameters"]
