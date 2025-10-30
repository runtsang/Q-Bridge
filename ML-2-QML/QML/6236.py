import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit that returns expectation values."""
    def __init__(self, backend: qiskit.providers.Provider, shots: int = 100):
        self._circuit = QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h([0, 1])
        self._circuit.barrier()
        self._circuit.ry(self.theta, 0)
        self._circuit.ry(self.theta, 1)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            z_vals = np.where(states == 0, 1, -1)
            return np.sum(z_vals * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a scalar through the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = circuit.run([val + shift])
            left = circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum hybrid layer that maps a scalar to a quantum expectation value."""
    def __init__(self, backend: qiskit.providers.Provider, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QuantumHybridEstimator:
    """
    Quantum estimator that evaluates a parametrised circuit and can be
    used as a hybrid head in a neural network.
    """
    def __init__(self,
                 backend: qiskit.providers.Provider,
                 shots: int = 100,
                 shift: float = np.pi/2):
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.circuit = QuantumCircuitWrapper(backend, shots)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circuit = self.circuit._circuit.assign_parameters({self.circuit.theta: params[0]})
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def hybrid_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the quantum circuit to each scalar in
        ``inputs`` and returns the expectation value.
        """
        return Hybrid(self.backend, self.shots, self.shift)(inputs)

__all__ = ["QuantumHybridEstimator", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
