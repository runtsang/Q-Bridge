"""
Quantum hybrid estimator combining variational feedforward, quantum convolution filter,
quantum self‑attention and graph‑based fidelity regularization.

It mirrors the classical EstimatorQNNGen275 but replaces neural network layers with
parameterized quantum circuits and replaces classical convolution with a quantum
convolution filter.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.random import random_circuit
import qutip
import networkx as nx
import itertools

# --- Quantum Convolution Filter ---
class QuantumConvFilter:
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 100
        # Build a reusable circuit template
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --- Quantum Self‑Attention ---
class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=self.shots)
        return job.result().get_counts(qc)

# --- Utility functions ---
def _random_unitary(num_qubits: int) -> qutip.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return qutip.Qobj(q)

def state_fidelity_q(a: qutip.Qobj, b: qutip.Qobj) -> float:
    return abs((a.dag() * b)[0,0]) ** 2

def fidelity_adjacency_q(states: list[qutip.Qobj], threshold: float,
                         *, secondary: float | None = None,
                         secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_q(s_i, s_j)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

# --- Main Quantum Estimator ---
class EstimatorQNNGen275:
    def __init__(self, arch: list[int], conv_kernel: int = 2, conv_threshold: float = 0.0,
                 embed_dim: int = 4, graph_threshold: float = 0.8):
        self.arch = arch
        self.conv = QuantumConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.graph_threshold = graph_threshold
        self.backend = Aer.get_backend("statevector_simulator")
        self.num_qubits = max(arch)
        self.layer_params = [np.random.rand(self.num_qubits) for _ in arch]
        self.observable = qutip.Pauli('Z')

    def _variational_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for i, p in enumerate(params):
            qc.ry(p, i)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def run(self, data):
        conv_out = self.conv.run(data)
        init_state = np.array([conv_out, 1 - conv_out])
        state = qutip.Qobj(init_state, dims=[[2],[1]])
        for params in self.layer_params:
            qc = self._variational_circuit(params)
            result = self.backend.run(qc).result()
            statevec = result.get_statevector()
            state = qutip.Qobj(statevec, dims=[[2]*self.num_qubits, [1]*self.num_qubits])
        rot_params = np.random.rand(3 * self.attention.n_qubits)
        ent_params = np.random.rand(self.attention.n_qubits - 1)
        attn_counts = self.attention.run(rot_params, ent_params)
        total = sum(int(k.count('1')) * v for k, v in attn_counts.items())
        attn_feat = total / (self.attention.n_qubits * self.attention.shots)
        exp_val = state.expectation_value(self.observable).real
        return exp_val * attn_feat

    def compute_graph(self, states: list[qutip.Qobj]) -> nx.Graph:
        return fidelity_adjacency_q(states, self.graph_threshold)

__all__ = ["EstimatorQNNGen275"]
