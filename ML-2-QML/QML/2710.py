import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Pauli
from collections.abc import Iterable, Sequence
from typing import List

class QuantumCircuitWrapper:
    """Parameterized circuit with entanglement and Z expectation measurement."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [QuantumCircuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        # Build circuit: H on all, RY(theta), CX entanglement, measure
        self.circuit.h(range(n_qubits))
        for i, th in enumerate(self.theta):
            self.circuit.ry(th, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter sets and return Z expectation."""
        if params.ndim == 1:
            params = params.reshape(1, -1)
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{th: val for th, val in zip(self.theta, row)} for row in params],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation of Z on first qubit
        exp = []
        for key, cnt in counts.items():
            z = 1 if key[0] == "0" else -1
            exp.append(z * cnt)
        exp = np.array(exp) / self.shots
        return exp

class QuantumHybridLayer(torch.autograd.Function):
    """Differentiable layer that forwards a scalar to a quantum circuit and returns its expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        params = inputs.detach().cpu().numpy()
        exp = circuit.run(params)
        result = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        params = inputs.detach().cpu().numpy()
        grad = []
        for val in params:
            exp_plus = circuit.run(np.array([val + shift]))
            exp_minus = circuit.run(np.array([val - shift]))
            grad.append((exp_plus - exp_minus) / (2 * np.sin(shift)))
        grad = torch.tensor(grad, dtype=torch.float32, device=inputs.device)
        return grad * grad_output, None, None

class FastHybridEstimator:
    """Efficient estimator for expectation values of a parameterised circuit."""
    def __init__(self, circuit: QuantumCircuitWrapper, shift: float = np.pi / 2, shots: int = 100, noise: bool = False):
        self.circuit = circuit
        self.shift = shift
        self.shots = shots
        self.noise = noise

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(
                self.circuit.circuit.assign_parameters(
                    {th: val for th, val in zip(self.circuit.theta, params)}
                )
            )
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if self.noise:
            rng = np.random.default_rng()
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

class HybridQCNet:
    """Quantum‑only wrapper that evaluates the hybrid circuit for a batch of inputs."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2, noise: bool = False):
        self.circuit = QuantumCircuitWrapper(n_qubits, backend or AerSimulator(), shots)
        self.estimator = FastHybridEstimator(self.circuit, shift, shots, noise)

    def evaluate(self, parameters: Sequence[Sequence[float]]) -> List[List[float]]:
        """Return the probability outputs for a batch of scalar parameters."""
        observable = Pauli("ZI")
        exp_values = self.estimator.evaluate([observable], parameters)
        probs = [float(0.5 * (1 + exp[0])) for exp in exp_values]
        return [[p, 1 - p] for p in probs]

__all__ = ["QuantumCircuitWrapper", "QuantumHybridLayer", "FastHybridEstimator", "HybridQCNet"]
