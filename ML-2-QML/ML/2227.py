import torch
import numpy as np
import networkx as nx
import itertools
from torch import nn, optim

__all__ = ["UnifiedQuantumGraphLayer"]

def random_network(arch, seed=None):
    rng = np.random.default_rng(seed)
    weights = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        w = torch.tensor(rng.standard_normal((out_f, in_f)), dtype=torch.float32, requires_grad=True)
        weights.append(w)
    return weights

def random_training_data(target_weight, samples, seed=None):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(samples):
        x = torch.tensor(rng.standard_normal(target_weight.shape[1]), dtype=torch.float32)
        y = target_weight @ x
        data.append((x, y))
    return data

def fidelity_adjacency(states, threshold, *, secondary=None, secondary_weight=0.5):
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = torch.dot(a, b) / (torch.norm(a)+1e-12) / (torch.norm(b)+1e-12)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class UnifiedQuantumGraphLayer(nn.Module):
    def __init__(self, arch, use_torch=True):
        super().__init__()
        self.arch = tuple(arch)
        self.use_torch = use_torch
        self.weights = nn.ParameterList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            w = nn.Parameter(torch.randn(out_f, in_f))
            self.weights.append(w)

    def forward(self, x):
        out = x
        for w in self.weights:
            out = torch.tanh(w @ out)
        return out

    def train(self, data, lr=0.01, epochs=100):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

    def to_qiskit(self, shots=1024):
        from qiskit import QuantumCircuit, Aer, execute
        n_qubits = self.arch[0]
        qc = QuantumCircuit(n_qubits)
        for layer_idx, w in enumerate(self.weights):
            for q in range(n_qubits):
                theta = torch.sum(w[:, q]).item()
                qc.ry(theta, q)
            if layer_idx < len(self.weights)-1:
                qc.barrier()
        qc.measure_all()
        return qc
