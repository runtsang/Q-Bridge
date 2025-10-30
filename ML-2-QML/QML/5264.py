"""Hybrid kernel, transformer, graph, classifier module with quantum components.

This module implements the same API as the classical counterpart but replaces
kernel, transformer, and classifier with quantum or hybrid versions.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
import itertools
import networkx as nx
from typing import Sequence, Iterable, Tuple


# --------------------------------------------------------------------------- #
#   Kernel implementations
# --------------------------------------------------------------------------- #

class RBFKernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a simple TX‑RX ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Encode with RX gates, then inverse for y
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
        )

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)
        self.encoder(self.q_device, -y)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
#   Transformer block (quantum)
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(tq.QuantumModule):
    """Transformer block that maps projections through quantum modules."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate, wire in zip(self.params, range(len(self.params))):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 8, q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.n_wires)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self.QLayer(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Quantum feed‑forward on each token
        q_outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            q_outputs.append(self.qlayer(token, qdev))
        q_out = torch.stack(q_outputs, dim=1)
        ffn_out = self.ffn(q_out)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#   Graph utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Absolute squared overlap between two pure states."""
    return abs((a.data.conj().T @ b.data)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#   Quantum classifier
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit,
                                                                    Iterable,
                                                                    Iterable,
                                                                    list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
#   Hybrid model
# --------------------------------------------------------------------------- #

class HybridKernelClassifier(nn.Module):
    """
    Hybrid model that combines classical/quantum kernels, transformer blocks,
    graph-based state fidelity, and a quantum or classical classifier.
    """

    def __init__(self,
                 kernel_type: str = 'rbf',
                 gamma: float = 1.0,
                 n_wires: int = 4,
                 transformer_cfg: dict | None = None,
                 graph_threshold: float = 0.9,
                 classifier_type: str = 'quantum',
                 classifier_depth: int = 3,
                 num_features: int | None = None) -> None:
        super().__init__()
        self.kernel_type = kernel_type
        self.graph_threshold = graph_threshold

        # Kernel
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(gamma)
        else:
            self.kernel = QuantumKernel(n_wires)

        # Transformer
        if transformer_cfg is None:
            transformer_cfg = dict(embed_dim=64, num_heads=4, dropout=0.1)
        self.transformer = TransformerBlockQuantum(**transformer_cfg, n_wires=n_wires)

        # Classifier
        embed_dim = transformer_cfg['embed_dim']
        if classifier_type == 'classical':
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
                num_qubits=embed_dim, depth=classifier_depth)

        self.num_features = num_features or embed_dim

    def _quantum_classify(self, repr: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum classifier on a batch of representations."""
        batch = repr.shape[0]
        logits = torch.zeros(batch, 2, device=repr.device)
        for i in range(batch):
            param_vals = repr[i].cpu().numpy()
            circ = self.classifier_circuit.copy()
            circ.assign_parameters({p: val for p, val in zip(self.encoding, param_vals)}, inplace=True)
            sv = Statevector.from_instruction(circ)
            exp = [sv.expectation_value(obs).real for obs in self.observables]
            logits[i] = torch.tensor(exp, dtype=torch.float32, device=repr.device)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, features).

        Returns
        -------
        torch.Tensor
            Logits for binary classification.
        """
        # Compute kernel matrix between input and itself
        batch = x.shape[0]
        kernel_matrix = torch.zeros(batch, batch, device=x.device)
        for i in range(batch):
            for j in range(batch):
                kernel_matrix[i, j] = self.kernel(x[i].unsqueeze(0), x[j].unsqueeze(0))

        # Treat kernel as sequence of embeddings
        embed_dim = self.transformer.norm1.normalized_shape[0]
        seq = kernel_matrix.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Transformer encoding
        h = self.transformer(seq)

        # Graph adjacency from state fidelity
        states = [Statevector.from_instruction(QuantumCircuit.from_qobj(q.to_qobj()))
                  for q in h.detach().cpu()]  # placeholder conversion
        _ = fidelity_adjacency(states, self.graph_threshold)

        # For demonstration, use mean‑pooled representation
        pooled = h.mean(dim=1)  # shape (batch, embed_dim)

        # Classifier
        if hasattr(self, 'classifier_circuit'):
            logits = self._quantum_classify(pooled)
        else:
            logits = self.classifier(pooled)

        return logits


__all__ = ["HybridKernelClassifier", "RBFKernel", "QuantumKernel",
           "TransformerBlockQuantum", "state_fidelity", "fidelity_adjacency",
           "build_classifier_circuit"]
