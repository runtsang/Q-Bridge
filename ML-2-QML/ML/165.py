import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble

class QuantumCircuitWrapper:
    """
    Variational two‑qubit circuit used in the hybrid head.
    The circuit applies Hadamard gates to both qubits, a
    parameterized Ry rotation on each qubit, and measures
    the expectation value of the Z observable.
    """
    def __init__(self, backend, shots=1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()

    def run(self, params):
        """
        Execute the circuit for a batch of parameters.
        :param params: numpy array of shape (batch, 2)
        :return: numpy array of expectation values of shape (batch,)
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: [p[0], p[1]]} for p in params]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        exp_vals = []
        for counts in result.get_counts():
            exp = 0.0
            for bitstring, cnt in counts.items():
                exp += int(bitstring, 2) * cnt
            exp /= self.shots
            exp_vals.append(exp)
        return np.array(exp_vals)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards the input through the quantum circuit
    and implements the parameter‑shift rule in the backward pass.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        np_inputs = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(np_inputs)
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        np_inputs = inputs.detach().cpu().numpy()
        grad_inputs = []
        for i in range(inputs.shape[0]):
            pos = np_inputs[i] + shift
            neg = np_inputs[i] - shift
            exp_pos = ctx.circuit.run(pos.reshape(1, -1))
            exp_neg = ctx.circuit.run(neg.reshape(1, -1))
            grad = (exp_pos - exp_neg) / 2.0
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(np.array(grad_inputs), dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output.unsqueeze(1), None, None

class QuantumHybridBinaryClassifier(nn.Module):
    """
    Hybrid classical‑quantum binary classifier.
    The model consists of a convolutional feature extractor,
    a fully‑connected head, and a quantum expectation layer
    that produces the final probability. The quantum layer
    uses a learnable shift parameter to control the finite‑difference
    step in the parameter‑shift rule.
    """
    def __init__(self, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, 2)
        self.shift = shift
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuitWrapper(self.backend, shots=shots)

    def forward(self, x: torch.Tensor):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.fc(feats)
        q_out = HybridFunction.apply(logits, self.quantum_circuit, self.shift)
        probs = torch.sigmoid(q_out)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridBinaryClassifier"]
