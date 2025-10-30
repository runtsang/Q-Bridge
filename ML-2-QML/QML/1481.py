import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit as QC, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class EmbeddingNet(nn.Module):
    """Learned feature extractor that reduces high-dimensional convolutional output to a compact embedding."""
    def __init__(self, in_features: int, embed_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.fc(x)))

class MultiQubitCircuit:
    """A parametrised circuit with entanglement and single-qubit rotations."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        theta = Parameter("theta")
        self.circuit = QC(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        for i in range(n_qubits):
            self.circuit.ry(theta, i)
        self.circuit.measure_all()
        self.compiled = transpile(self.circuit, self.backend)
    def run(self, params: np.ndarray) -> np.ndarray:
        bindings = [{self.circuit.parameters[0]: p} for p in params]
        qobj = assemble(self.compiled, shots=self.shots, parameter_binds=bindings)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            probs = counts / self.shots
            states = np.array([int(s, 2) for s in count_dict.keys()])
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: MultiQubitCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = torch.tensor([circuit.run(inputs[i].tolist())[0] for i in range(inputs.shape[0])])
        ctx.save_for_backward(inputs)
        return expectations
    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            for q in range(circuit.n_qubits):
                params_plus = inputs[i].clone().detach().numpy()
                params_minus = params_plus.copy()
                params_plus[q] += shift
                params_minus[q] -= shift
                exp_plus = circuit.run(params_plus)[0]
                exp_minus = circuit.run(params_minus)[0]
                grad_inputs[i, q] = (exp_plus - exp_minus) / 2
        return grad_inputs * grad_output.unsqueeze(1), None, None
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        if backend is None:
            backend = AerSimulator()
        self.circuit = MultiQubitCircuit(n_qubits, backend, shots)
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)
class QuantumHybridClassifier(nn.Module):
    """Convolutional network followed by a multi-qubit quantum expectation head."""
    def __init__(self, conv_backbone: nn.Module, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = conv_backbone
        self.embed = EmbeddingNet(self.backbone.out_features, embed_dim=8)
        self.hybrid = Hybrid(n_qubits, shots=1024, shift=shift)
        self.output_linear = nn.Linear(8, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        flat = torch.flatten(feats, 1)
        emb = self.embed(flat)
        params = emb[:, :self.hybrid.circuit.n_qubits]
        qs = self.hybrid(params)
        logits = self.output_linear(emb)
        probs = torch.sigmoid(logits + qs.unsqueeze(1))
        return torch.cat((probs, 1 - probs), dim=-1)
