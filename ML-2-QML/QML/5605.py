import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import numpy as np
import networkx as nx

class QuantumCircuit:
    """Simple two‑qubit variational circuit with a single rotation parameter."""
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        for q in range(n_qubits):
            self.circuit.h(q)
            self.circuit.ry(self.theta, q)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas):
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for outcome, count in counts.items():
            z = int(outcome[::-1].replace('0', '-1').replace('1', '1'))
            exp += z * count
        return np.array([exp / self.shots])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        exp = circuit.run(thetas)
        result = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = ctx.circuit.run([val + shift])
            exp_minus = ctx.circuit.run([val - shift])
            grads.append((exp_plus - exp_minus) / (2 * shift))
        grads = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grads * grad_output, None, None

class HybridLayer(nn.Module):
    def __init__(self, n_qubits, backend, shots, shift):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x):
        return HybridFunction.apply(x, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN backbone followed by a quantum expectation head and graph aggregation."""
    def __init__(self, adjacency_threshold=0.9, shift=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.hybrid = HybridLayer(1, backend, shots=1024, shift=shift)
        self.adjacency_threshold = adjacency_threshold

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
        features = self.fc2(x)
        graph = self._fidelity_adjacency(features, self.adjacency_threshold)
        x = self.fc3(features)
        x = self.hybrid(x)
        probs = torch.sigmoid(x).squeeze(-1)
        agg = probs.clone()
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                agg[node] += probs[neighbors].mean()
            deg = graph.degree[node]
            agg[node] = agg[node] / (1 + deg)
        return torch.stack([agg, 1 - agg], dim=-1)

    @staticmethod
    def _fidelity_adjacency(states, threshold):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i+1:], i+1):
                fid = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph

__all__ = ["HybridBinaryClassifier"]
