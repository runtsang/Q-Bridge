import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt
import networkx as nx
import numpy as np
import itertools

__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "generate_superposition_data", "fidelity_adjacency"]

def build_classifier_circuit(num_qubits: int, depth: int):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def generate_superposition_data(num_wires: int, samples: int):
    omega_0 = qt.basis(2 ** num_wires, 0)
    omega_1 = qt.basis(2 ** num_wires, -1)
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = []
    labels = []
    for theta, phi in zip(thetas, phis):
        state = np.cos(theta) * omega_0 + np.exp(1j * phi) * np.sin(theta) * omega_1
        states.append(state)
        labels.append(np.sin(2 * theta) * np.cos(phi))
    return states, np.array(labels, dtype=np.float32)

def fidelity_adjacency(states: list[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = abs((a.dag() * b)[0,0]) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class QuantumCircuitEncoder(tq.QuantumModule):
    def __init__(self, num_qubits: int, num_features: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(num_features)]
        )
        self.var_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.var_layer(qdev)
        return self.measure(qdev)

class QuantumTransformerBlock(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_wires = max(embed_dim, num_heads)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(embed_dim)]
        )
        self.q_attention = tq.RandomLayer(n_ops=10, wires=list(range(self.n_wires)))
        self.q_feedforward = tq.RandomLayer(n_ops=10, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.q_attention(qdev)
        attn_out = self.measure(qdev)
        self.q_feedforward(qdev)
        ffn_out = self.measure(qdev)
        out = attn_out + ffn_out
        out = self.dropout(out)
        return self.linear(out)

class QuantumGraphEncoder(tq.QuantumModule):
    def __init__(self, num_nodes: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i % n_qubits]} for i in range(num_nodes)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        return self.measure(qdev)

class QuantumClassifierModel(tq.QuantumModule):
    def __init__(self,
                 data_type: str = "tabular",
                 num_features: int | None = None,
                 vocab_size: int | None = None,
                 num_classes: int = 2,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 n_qubits: int = 4,
                 regression: bool = False):
        super().__init__()
        self.data_type = data_type
        self.regression = regression
        self.num_classes = num_classes if not regression else 1

        if data_type == "tabular":
            self.encoder = QuantumCircuitEncoder(num_qubits=n_qubits,
                                                 num_features=num_features)
            self.head = nn.Linear(n_qubits, num_classes if not regression else 1)
        elif data_type == "text":
            self.transformers = nn.Sequential(
                *[QuantumTransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
            )
            self.head = nn.Linear(embed_dim, num_classes if not regression else 1)
        elif data_type == "graph":
            self.encoder = QuantumGraphEncoder(num_nodes=n_qubits, n_qubits=n_qubits)
            self.head = nn.Linear(n_qubits, num_classes if not regression else 1)
        elif data_type == "quantum":
            self.encoder = QuantumCircuitEncoder(num_qubits=n_qubits,
                                                 num_features=num_features)
            self.head = nn.Linear(n_qubits, num_classes if not regression else 1)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.data_type == "tabular":
            qdev = tq.QuantumDevice(n_wires=self.encoder.num_qubits,
                                    bsz=state_batch.shape[0],
                                    device=state_batch.device)
            self.encoder.encoder(qdev, state_batch)
            self.encoder.var_layer(qdev)
            features = self.encoder.measure(qdev)
            return self.head(features)
        elif self.data_type == "text":
            out = state_batch
            for block in self.transformers:
                qdev = tq.QuantumDevice(n_wires=block.n_wires,
                                        bsz=out.shape[0],
                                        device=out.device)
                out = block(qdev)
            out = out.mean(dim=1)
            return self.head(out)
        elif self.data_type == "graph":
            qdev = tq.QuantumDevice(n_wires=self.encoder.n_qubits,
                                    bsz=state_batch.shape[0],
                                    device=state_batch.device)
            self.encoder.forward(qdev)
            features = self.encoder.measure(qdev)
            return self.head(features)
        elif self.data_type == "quantum":
            qdev = tq.QuantumDevice(n_wires=self.encoder.num_qubits,
                                    bsz=state_batch.shape[0],
                                    device=state_batch.device)
            self.encoder.encoder(qdev, state_batch)
            self.encoder.var_layer(qdev)
            features = self.encoder.measure(qdev)
            return self.head(features)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
