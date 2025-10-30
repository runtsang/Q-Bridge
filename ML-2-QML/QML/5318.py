"""Quantum component for FraudDetection – a variational two‑qubit circuit.

The quantum module mirrors the design in *QuantumCircuit* from
ClassicalQuantumBinaryClassification but is adapted for integration
with the classical FraudDetection model.  It provides:

1. **QuantumCircuit** – a parametrised two‑qubit circuit that
   accepts a single angle per sample and returns the expectation
   value of the Z‑observable.

2. **HybridFunction** – a PyTorch autograd function that forwards
   the classical logits to the quantum circuit and back‑propagates
   via finite differences.  This follows the pattern from
   *HybridFunction* in the reference pair.

3. **Hybrid** – a PyTorch module that exposes the quantum circuit
   as a differentiable layer.

The implementation uses Qiskit Aer for simulation and is
fully compatible with the classical FraudDetection model defined
in the ML module.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import assemble, transpile
from qiskit.circuit import Parameter, QuantumCircuit as QC
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector, Operator

# ----------------------------------------------------------------------
# 1. Quantum circuit wrapper
# ----------------------------------------------------------------------
class QuantumCircuit:
    """
    Two‑qubit circuit with a parametrised Ry gate.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2).
    backend : Backend
        Qiskit backend for execution.
    shots : int
        Number of shots per evaluation.
    """

    def __init__(self, n_qubits: int, backend: Backend, shots: int) -> None:
        self._circuit = QC(n_qubits)
        self.theta = Parameter("θ")
        all_qubits = list(range(n_qubits))

        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict[str, int]) -> float:
            counts = np.array(list(count_dict.values()), dtype=float)
            states = np.array([int(k, 2) for k in count_dict.keys()], dtype=float)
            probs = counts / self.shots
            return float(np.sum(states * probs))

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


# ----------------------------------------------------------------------
# 2. Differentiable interface
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """
    Forward a scalar to the quantum circuit and back‑propagate via
    central finite differences.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        # Run the circuit on the CPU (PyTorch tensors are converted to lists)
        angles = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(angles)
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        # Central difference
        angles_plus = (inputs + shift).detach().cpu().numpy()
        angles_minus = (inputs - shift).detach().cpu().numpy()
        exp_plus = ctx.circuit.run(angles_plus)
        exp_minus = ctx.circuit.run(angles_minus)
        gradients = (exp_plus - exp_minus) / (2 * shift)
        grad = torch.tensor(gradients, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None


# ----------------------------------------------------------------------
# 3. Hybrid layer
# ----------------------------------------------------------------------
class Hybrid(nn.Module):
    """
    PyTorch module that forwards activations through a quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : Backend
        Qiskit backend.
    shots : int
        Shot count for simulation.
    shift : float
        Finite‑difference shift value.
    """

    def __init__(self, n_qubits: int, backend: Backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.quantum_circuit, self.shift)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid"]
