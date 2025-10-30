"""Pure quantum binary classifier using a 2‑qubit sampler network."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
import qiskit_aer


class SamplerQNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, sampler: QiskitSampler, circuit: qiskit.QuantumCircuit):
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


class HybridBinaryClassifier(nn.Module):
    """Standalone quantum binary classifier based on a 2‑qubit sampler network."""
    def __init__(self) -> None:
        super().__init__()
        input_params = ParameterVector("input", 2)
        weight_params = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)
        self.circuit = qc
        self.sampler = QiskitSampler()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return SamplerQNNFunction.apply(inputs, self.sampler, self.circuit)


__all__ = ["HybridBinaryClassifier"]
