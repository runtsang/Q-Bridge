"""Hybrid quantum‑classical fully‑connected layer with a differentiable quantum head.

The quantum part is a parameterised circuit executed on Qiskit’s Aer simulator.
A custom autograd function bridges PyTorch and the quantum backend, providing
finite‑difference gradients for back‑propagation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuantumCircuitWrapper:
    """
    Parameterised circuit that applies an RY rotation to each qubit.
    The circuit can be executed on a chosen Qiskit backend.
    """

    def __init__(self, n_qubits: int, backend, shots: int = 100) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Execute the circuit for each angle and return the expectation of Z."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: angle} for angle in angles],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(cnt) for cnt in result])
        return np.array([expectation(result)])


class QuantumHybridFunction(Function):
    """
    PyTorch autograd function that forwards activations through a quantum circuit.
    Finite‑difference gradients are computed for the backward pass.
    """

    @staticmethod
    def forward(ctx: "QuantumHybridFunction", inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert inputs to a 1‑D numpy array of angles
        angles = inputs.detach().cpu().numpy()
        expectation = ctx.circuit.run(angles)
        return torch.tensor(expectation, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx: "QuantumHybridFunction", grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inputs = grad_output.detach().cpu().numpy()
        shift = ctx.shift
        gradients = []
        for angle in inputs:
            right = ctx.circuit.run([angle + shift])[0]
            left = ctx.circuit.run([angle - shift])[0]
            gradients.append(right - left)
        grad_inputs = torch.tensor(gradients, dtype=torch.float32, device=grad_output.device) * grad_output
        return grad_inputs, None, None


class HybridQuantumLayer(nn.Module):
    """
    Layer that forwards its input through a quantum circuit.
    """

    def __init__(self, n_qubits: int, backend, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)


class HybridFullyConnectedLayerQuantum(nn.Module):
    """
    Hybrid network that mirrors the classical fully‑connected layer but ends with a quantum expectation head.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] = [32, 16],
        n_qubits: int = 1,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        layers = []
        prev = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_qubits))
        self.net = nn.Sequential(*layers)
        self.quantum_layer = HybridQuantumLayer(n_qubits, backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.net(x)
        return self.quantum_layer(features)


# Example of a quantum estimator‑based regressor
def estimator_qnn_example() -> EstimatorQNN:
    """
    Build a simple Qiskit EstimatorQNN that maps a single input to a single output.
    """
    input_param = Parameter("input")
    weight_param = Parameter("weight")
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.ry(input_param, 0)
    qc.rx(weight_param, 0)

    observable = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=[weight_param],
        estimator=estimator,
    )


__all__ = [
    "QuantumCircuitWrapper",
    "QuantumHybridFunction",
    "HybridQuantumLayer",
    "HybridFullyConnectedLayerQuantum",
    "estimator_qnn_example",
]
