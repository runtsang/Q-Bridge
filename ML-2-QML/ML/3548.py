"""Hybrid classical-quantum binary classifier with expectation and sampler heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import qiskit
from qiskit import transpile, assemble
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler as QiskitSampler
import qiskit_aer


class QuantumCircuit:
    """Parametrised single‑qubit circuit used for expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
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
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable expectation head via finite‑difference."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().numpy()) * ctx.shift
        grad_inputs = []
        for idx, val in enumerate(inputs.detach().numpy()):
            grad_right = ctx.circuit.run([val + shift[idx]])[0]
            grad_left = ctx.circuit.run([val - shift[idx]])[0]
            grad_inputs.append((grad_right - grad_left) / (2 * shift[idx]))
        grad_tensor = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad_tensor * grad_output, None, None


class SamplerFunction(torch.autograd.Function):
    """Differentiable sampler head using Qiskit’s StatevectorSampler."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, sampler: QiskitSampler, circuit: qiskit.QuantumCircuit) -> torch.Tensor:
        ctx.sampler = sampler
        ctx.circuit = circuit
        probs = []
        for sample in inputs.tolist():
            counts = ctx.sampler.sample(circuit, parameter_values=sample)
            probs_dict = {k: v / sum(counts.values()) for k, v in counts.items()}
            p0 = probs_dict.get("00", 0.0) + probs_dict.get("01", 0.0)
            p1 = probs_dict.get("10", 0.0) + probs_dict.get("11", 0.0)
            probs.append([p0, p1])
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        ctx.save_for_backward(inputs, probs_tensor)
        return probs_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        sampler = ctx.sampler
        circuit = ctx.circuit
        shift = 1e-3
        grad_inputs = []
        for i, sample in enumerate(inputs.tolist()):
            grad_sample = []
            for j, val in enumerate(sample):
                perturbed_plus = sample.copy()
                perturbed_plus[j] += shift
                perturbed_minus = sample.copy()
                perturbed_minus[j] -= shift
                probs_plus = sampler.sample(circuit, parameter_values=perturbed_plus)
                probs_minus = sampler.sample(circuit, parameter_values=perturbed_minus)
                probs_plus = {k: v / sum(probs_plus.values()) for k, v in probs_plus.items()}
                probs_minus = {k: v / sum(probs_minus.values()) for k, v in probs_minus.items()}
                p0_plus = probs_plus.get("00", 0.0) + probs_plus.get("01", 0.0)
                p1_plus = probs_plus.get("10", 0.0) + probs_plus.get("11", 0.0)
                p0_minus = probs_minus.get("00", 0.0) + probs_minus.get("01", 0.0)
                p1_minus = probs_minus.get("10", 0.0) + probs_minus.get("11", 0.0)
                grad_p0 = (p0_plus - p0_minus) / (2 * shift)
                grad_p1 = (p1_plus - p1_minus) / (2 * shift)
                grad_sample.append([grad_p0, grad_p1])
            grad_inputs.append(grad_sample)
        grad_tensor = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad_tensor * grad_output, None, None


class Hybrid(nn.Module):
    """Wrapper that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)


class HybridBinaryClassifier(nn.Module):
    """CNN backbone with two quantum heads: expectation and sampler."""
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend = qiskit_aer.get_backend("aer_simulator")
        self.expectation_head = Hybrid(1, backend, shots=100, shift=shift)

        self.sampler_backend = qiskit_aer.get_backend("aer_simulator")
        self.sampler = QiskitSampler()
        input_params = qiskit.circuit.ParameterVector("input", 2)
        weight_params = qiskit.circuit.ParameterVector("weight", 4)
        qc = qiskit.QuantumCircuit(2)
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)
        self.sampler_circuit = qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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

        exp_logits = self.expectation_head(x.squeeze(-1))
        exp_prob = torch.sigmoid(exp_logits)

        sampler_input = torch.cat([x, x], dim=-1).squeeze(-1)
        samp_probs = SamplerFunction.apply(sampler_input, self.sampler, self.sampler_circuit)

        combined = 0.5 * exp_prob + 0.5 * samp_probs[:, 1]
        final_prob = combined
        return torch.cat((final_prob, 1 - final_prob), dim=-1)


__all__ = ["HybridBinaryClassifier"]
