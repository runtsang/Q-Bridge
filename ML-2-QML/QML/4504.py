"""Hybrid quantum self‑attention module combining quantum attention, regression, graph, and sampler."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler
import qutip as qt
import networkx as nx
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Quantum utilities
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random unitary for `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Generate a random pure state for `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Produce input‑output pairs for a quantum regression task."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def fidelity_adjacency(states: list[qt.Qobj], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """
    Build a weighted graph where edges represent fidelity between quantum states.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = abs((states[i].dag() * states[j])[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 2. Hybrid quantum self‑attention class
# --------------------------------------------------------------------------- #

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention that mirrors the classical counterpart.
    Provides a parameterised attention circuit, a quantum regression block,
    graph‑based state propagation, and a Qiskit sampler network.
    """

    def __init__(self,
                 n_qubits: int,
                 graph_threshold: float = 0.8,
                 use_sampler: bool = False):
        self.n_qubits = n_qubits
        self.graph_threshold = graph_threshold
        self.use_sampler = use_sampler

        # Quantum attention parameters
        self.rotation_params = np.random.uniform(0, 2 * np.pi, size=3 * n_qubits)
        self.entangle_params = np.random.uniform(0, 2 * np.pi, size=n_qubits - 1)

        # Regression components
        self.qreg_model = self._build_qregression_model()
        self.qreg_dataset = self._build_qregression_dataset()

        # Sampler
        self.sampler = self._build_sampler() if use_sampler else None

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_qregression_model(self):
        """A minimal variational regression circuit followed by a linear read‑out."""
        class QRegressionModel(nn.Module):
            def __init__(self, n_qubits: int):
                super().__init__()
                self.n_qubits = n_qubits
                self.head = nn.Linear(n_qubits, 1)

            def forward(self, x: torch.Tensor):
                return self.head(x)
        return QRegressionModel(self.n_qubits)

    def _build_qregression_dataset(self):
        """Generate superposition states and labels for a regression task."""
        omega_0 = np.zeros(2 ** self.n_qubits, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** self.n_qubits, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(200)
        phis = 2 * np.pi * np.random.rand(200)
        states = np.zeros((200, 2 ** self.n_qubits), dtype=complex)
        labels = np.sin(2 * thetas) * np.cos(phis)
        for i in range(200):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        return list(zip(states, labels))

    def _build_sampler(self):
        """Instantiate a Qiskit neural network sampler."""
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
        return QSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def _build_circuit(self):
        """Construct the attention circuit used in `run_quantum_attention`."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(self.rotation_params[3 * i], i)
            circuit.ry(self.rotation_params[3 * i + 1], i)
            circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(self.entangle_params[i], i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def run_quantum_attention(self, inputs: np.ndarray, shots: int = 1024):
        """Execute the attention circuit and return measurement counts."""
        circuit = self._build_circuit()
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def build_graph(self, states: list[qt.Qobj]) -> nx.Graph:
        """Create a fidelity‑based adjacency graph from quantum states."""
        return fidelity_adjacency(states, self.graph_threshold)

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Return a probability distribution from the quantum sampler."""
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled.")
        inp = torch.tensor(inputs, dtype=torch.float32)
        return self.sampler(inp).detach().numpy()

    def train_qregression(self, epochs: int = 5, lr: float = 1e-3):
        """Placeholder training loop for the quantum regression block."""
        optimizer = torch.optim.Adam(self.qreg_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for state, target in self.qreg_dataset:
                state_tensor = torch.tensor(state, dtype=torch.cfloat)
                pred = self.qreg_model(state_tensor)
                loss = loss_fn(pred.squeeze(), torch.tensor(target, dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
