"""Hybrid quantum implementation of the same architecture.

It uses a qiskit circuit as a variational expectation layer and a classical
CNN backbone.  The quantum head is built by :func:`build_classifier_circuit`
which returns a parametrised circuit, encoding and observable list.  The
HybridFunction implements a differentiable interface that allows gradient
propagation through the circuit.

A lightweight :class:`FastEstimator` is also provided to evaluate the model
with optional shot noise, mirroring the classical version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Sequence, List, Callable

# --------------------------------------------------------------------------- #
# Utility: build_classifier_circuit (quantum variant)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[qiskit.QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised variational circuit.
    encoding : Iterable
        ParameterVector for data encoding.
    weights : Iterable
        ParameterVector for variational weights.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Thin wrapper around a qiskit circuit that can be evaluated on a backend."""

    def __init__(self, circuit: qiskit.QuantumCircuit, backend, shots: int):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.params = list(circuit.parameters)

    def run(self, param_values: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: v} for p, v in zip(self.params, param_values)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on the first qubit (single‑qubit ansatz)
        exp_val = 0.0
        for state, count in result.items():
            prob = count / self.shots
            # state is a bitstring; the first qubit is the most significant bit
            bit = int(state[0])
            exp_val += prob * (1 - 2 * bit)
        return np.array([exp_val])

# --------------------------------------------------------------------------- #
# Differentiable interface
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(exp, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grad = torch.tensor(grads, device=inputs.device, dtype=torch.float32)
        return grad * grad_output, None, None

# --------------------------------------------------------------------------- #
# Hybrid layer
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(n_qubits, depth=2)
        self.circuit = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a hybrid network for a batch of inputs and scalar observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic predictions."""

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# HybridQCNet (quantum)
# --------------------------------------------------------------------------- #
class HybridQCNet(nn.Module):
    """CNN backbone followed by a quantum expectation head."""

    def __init__(self, num_features: int = 3, n_qubits: int = 1, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).T
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNet", "FastEstimator", "FastBaseEstimator", "build_classifier_circuit"]
