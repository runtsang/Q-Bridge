"""Unified hybrid classifier with a variational quantum head.

The module implements a quantum‑augmented version of the classical
backbone.  The quantum circuit is a data‑re‑upload ansatz with
parameter‑shift gradient estimation, compatible with Qiskit Aer
simulator.  The returned `UnifiedHybridClassifier` class exposes the
same public API as the classical implementation so that model
selection is transparent.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit as QiskitCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QiskitCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational circuit with data re‑uploading.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
        The compiled variational ansatz.
    encoding : list[Parameter]
        The data‑encoding parameters.
    weights : list[Parameter]
        The variational parameters.
    observables : list[SparsePauliOp]
        Pauli Z observables on each qubit used to compute expectation
        values.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QiskitCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    idx = 0
    for _ in range(depth):
        # Rotation layer
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        # Entangling layer
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)
    # Measurement observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class QuantumCircuitWrapper:
    """
    Lightweight wrapper around a parametrised Qiskit circuit executed on
    Aer simulator.  The `run` method returns the expectation of the
    Pauli Z operator for each qubit.
    """

    def __init__(self, n_qubits: int, shots: int = 1024):
        self._circuit = QiskitCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        theta = QiskitCircuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.ry(theta, all_qubits)
        self._circuit.measure_all()

        self.backend = AerSimulator()
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self._circuit.parameters[0]: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge that evaluates the quantum circuit and provides
    a parameter‑shift gradient estimate.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit

        expectation_z = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation_z, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)

        grads = []
        for val, s in zip(inputs.detach().cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """
    Quantum expectation layer that can be plugged into a PyTorch model.
    """

    def __init__(self, n_qubits: int, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.quantum = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.quantum, self.shift)


class UnifiedHybridClassifier(nn.Module):
    """
    Residual classical backbone followed by a variational quantum expectation
    head.  The architecture matches the classical counterpart in depth and
    feature size, so that the two can be swapped without changing the
    training loop.
    """

    def __init__(self, num_features: int, depth: int = 3, n_qubits: int = 2, shift: float = np.pi / 2):
        super().__init__()
        self.classifier, _, _, _ = build_classifier_circuit(num_features, depth)
        self.hybrid = Hybrid(n_qubits, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        # Use the first output channel as the quantum input
        q_input = x[:, 0]
        q_output = self.hybrid(q_input)
        return torch.cat([q_output, 1 - q_output], dim=-1)


__all__ = ["build_classifier_circuit", "UnifiedHybridClassifier", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
