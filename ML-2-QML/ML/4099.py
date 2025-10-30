import torch
from torch import nn
import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

def EstimatorQNN() -> nn.Module:
    """Return a simple fully‑connected regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)
    return EstimatorNN()

class QuantumHead(nn.Module):
    """A Qiskit variational circuit with two parameters and a Y‑observable."""
    def __init__(self, shots: int = 1, seed: int = 42) -> None:
        super().__init__()
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(1)
        self.input_param = self.circuit.parameter("theta_in")
        self.weight_param = self.circuit.parameter("theta_w")
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.circuit.measure_all()
        self.compiled = transpile(self.circuit, self.backend)
        self.observable = SparsePauliOp.from_list([("Y", 1)])

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bind = {self.input_param: inputs.numpy(), self.weight_param: weights.numpy()}
        qobj = assemble(self.compiled, parameter_binds=[bind])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def _expect(counts):
            probs = np.array(list(counts.values())) / sum(counts.values())
            states = np.array(list(counts.keys()), dtype=float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return torch.tensor([_expect(item) for item in result])
        return torch.tensor([_expect(result)])

class HybridEstimatorQNN(nn.Module):
    """Hybrid estimator that merges a classical feed‑forward backbone with an optional quantum head."""
    def __init__(self, use_quantum: bool = True, use_transformer: bool = False, seed: int = 42) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.use_transformer = use_transformer
        self.backbone = EstimatorQNN()
        if use_transformer:
            self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
        if use_quantum:
            self.quantum_head = QuantumHead()
            self.weight = nn.Parameter(torch.randn(1))
        else:
            self.linear = nn.Linear(1, 1, bias=True)
            self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        if self.use_transformer:
            seq = x.unsqueeze(1)
            attn_out, _ = self.attn(seq, seq, seq)
            x = self.ffn(attn_out.squeeze(1))
        if self.use_quantum:
            out = self.quantum_head(x.squeeze(-1), self.weight)
        else:
            out = self.linear(x) + self.shift
        return out

def HybridEstimatorQNNFactory(use_quantum: bool = True, use_transformer: bool = False, seed: int = 42) -> nn.Module:
    """Return an instance of :class:`HybridEstimatorQNN`."""
    return HybridEstimatorQNN(use_quantum, use_transformer, seed)
