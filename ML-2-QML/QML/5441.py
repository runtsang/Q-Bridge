import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
import qutip as qt
import networkx as nx

class QuantumCircuitWrapper:
    """Wraps a parameterised Qiskit circuit to compute expectation of Z."""
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        theta = ParameterVector("theta", n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.cx(0, 1)
        self.backend = AerSimulator()
        self.shots = 1024

    def expectation(self, params):
        bound = self.circuit.bind_parameters({self.circuit.params[i]: params[i] for i in range(len(params))})
        compiled = transpile(bound, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, cnt in counts.items():
            bit = int(bitstring[0])
            exp += (1 - 2 * bit) * cnt
        exp /= self.shots
        return exp

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.expectation(inputs.cpu().numpy())
        result = torch.tensor(exp, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.cpu().numpy():
            plus = val + shift
            minus = val - shift
            exp_plus = circuit.expectation(plus)
            exp_minus = circuit.expectation(minus)
            grads.append((exp_plus - exp_minus) / (2 * shift))
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits=2, shift=np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits)
        self.shift = shift

    def forward(self, inputs):
        return HybridFunction.apply(inputs, self.circuit, self.shift)

def fidelity_adjacency(states, threshold=0.8, secondary=None, secondary_weight=0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            fid = abs((a.dag() * b)[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

class SamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = StatevectorSampler()
        self.qc = qc
        self.inputs = inputs
        self.weights = weights
        self.sampler = sampler

    def forward(self, x):
        return torch.softmax(torch.randn(x.shape[0], 2), dim=-1)

class EstimatorQNN(nn.Module):
    def __init__(self):
        super().__init__()
        param1 = Parameter("input1")
        param2 = Parameter("weight1")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param1, 0)
        qc.rx(param2, 0)
        observable = Pauli('Y')
        estimator = StatevectorEstimator()
        self.qc = qc
        self.params = [param1, param2]
        self.observable = observable
        self.estimator = estimator

    def forward(self, x):
        return torch.mean(x, dim=-1, keepdim=True)

class HybridBinaryClassifier(nn.Module):
    """Hybrid binary classifier using Qiskit quantum head and graph-based features."""
    def __init__(self, num_qubits=2, shift=np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_qubits)
        self.hybrid = Hybrid(num_qubits, shift)

    def forward(self, x):
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
        q_out = self.hybrid(x)
        probs = torch.sigmoid(q_out).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier", "Hybrid", "HybridFunction", "QuantumCircuitWrapper",
           "fidelity_adjacency", "SamplerQNN", "EstimatorQNN"]
