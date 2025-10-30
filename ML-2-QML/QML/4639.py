import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Tuple
import qiskit
from qiskit import assemble, transpile

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑style quantum layer."""
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

def _apply_layer(qc: qiskit.QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    # Simplified photonic layer using standard gates
    for i in range(qc.num_qubits):
        qc.h(i)
    qc.ry(params.bs_theta, range(qc.num_qubits))
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.s(i)
        qc.u3(r if not clip else _clip(r, 5.0), phi, 0.0, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.cx(i, (i + 1) % qc.num_qubits)
    for i, k in enumerate(params.kerr):
        qc.rz(k if not clip else _clip(k, 1.0), i)

def build_quantum_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    n_qubits: int,
) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(n_qubits)
    _apply_layer(qc, input_params, clip=False)
    for l in layers:
        _apply_layer(qc, l, clip=True)
    qc.measure_all()
    return qc

class HybridQuantumBinaryClassifier(nn.Module):
    """Quantum hybrid classifier that evaluates a parameterised circuit
    built from :class:`FraudLayerParameters`.  Gradient is computed via
    the parameter‑shift rule and the output is passed through a sigmoid."""
    def __init__(
        self,
        n_qubits: int,
        backend,
        shots: int,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        # Default parameters; in practice user supplies own
        self.input_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.layers: Tuple[FraudLayerParameters] = ()
        self.circuit = build_quantum_program(self.input_params, self.layers, n_qubits)

    def expectation(self, theta: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.circuit.parameters[0]: t} for t in theta])
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=float)
        return np.sum(states * probs)

    def parameter_shift(self, theta: np.ndarray) -> np.ndarray:
        eps = self.shift
        return (self.expectation(theta + eps) - self.expectation(theta - eps)) / (2 * np.sin(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        theta = x.squeeze().detach().cpu().numpy()
        exp_val = self.parameter_shift(theta)
        probs = torch.sigmoid(torch.tensor(exp_val, device=x.device))
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier", "FraudLayerParameters", "build_quantum_program"]
