"""Quantum head for the hybrid EstimatorQNNGen297 model.

The quantum circuit is built using a layered ansatz with data‑encoding
and variational parameters.  A differentiable PyTorch wrapper
(`HybridFunction`) allows training of the full hybrid model with
automatic back‑propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend

# --------------------------------------------------------------------
# Build a layered classifier circuit (from reference 4)
# --------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """Construct a simple variational ansatz with explicit encoding and
    variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------
# Quantum circuit executor
# --------------------------------------------------------------------
class QuantumCircuitExecutor:
    """Runs a parametrised circuit on a backend and returns expectation
    values for the given observables."""
    def __init__(self, circuit: QuantumCircuit, backend: Backend, shots: int) -> None:
        self._circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: val} for p, val in zip(self._circuit.parameters, params)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()]).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------
# Differentiable bridge between PyTorch and the quantum circuit
# --------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, executor: QuantumCircuitExecutor, observables: list[SparsePauliOp], shift: float):
        ctx.shift = shift
        ctx.executor = executor
        ctx.observables = observables

        # Forward pass
        param_vals = inputs.detach().cpu().numpy()
        expectations = ctx.executor.run(param_vals)
        output = torch.tensor(expectations, device=inputs.device)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for idx in range(inputs.shape[1]):
            right = ctx.executor.run((inputs + shift).detach().cpu().numpy())
            left = ctx.executor.run((inputs - shift).detach().cpu().numpy())
            grads.append(right - left)
        grads = torch.tensor(grads, device=inputs.device).float()
        return grads * grad_output, None, None, None

# --------------------------------------------------------------------
# Quantum hybrid layer (from reference 3)
# --------------------------------------------------------------------
class EstimatorQNNGen297(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, num_qubits: int = 2, depth: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        circuit, encoding, weights, observables = build_classifier_circuit(num_qubits, depth)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.executor = QuantumCircuitExecutor(circuit, backend, shots)
        self.observables = observables
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs: (batch, num_qubits)
        return HybridFunction.apply(inputs, self.executor, self.observables, self.shift)
