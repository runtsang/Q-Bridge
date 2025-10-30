import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Optional

class ConvGraphQNN:
    """
    Quantum‑enhanced convolution + graph neural network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        conv_threshold: float = 0.0,
        graph_threshold: float = 0.9,
        secondary_threshold: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _run_patch(self, patch: np.ndarray) -> float:
        """
        Execute the quantum circuit for a single patch and return
        the average probability of measuring |1> across qubits.
        """
        param_binds = []
        for val in patch.flatten():
            bind = {}
            for i, v in enumerate(patch.flatten()):
                bind[self.theta[i]] = np.pi if v > self.conv_threshold else 0.0
            param_binds.append(bind)
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = 0
        for outcome, count in counts.items():
            ones = outcome.count("1")
            total += ones * count
        prob = total / (self.shots * self.n_qubits)
        return prob

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for all patches of the batch by running the quantum circuit.
        """
        B, C, H, W = x.shape
        probs = torch.empty(B, (H - self.kernel_size + 1) * (W - self.kernel_size + 1), dtype=torch.float32)
        for b in range(B):
            img = x[b, 0].cpu().numpy()
            idx = 0
            for i in range(H - self.kernel_size + 1):
                for j in range(W - self.kernel_size + 1):
                    patch = img[i : i + self.kernel_size, j : j + self.kernel_size]
                    probs[b, idx] = self._run_patch(patch)
                    idx += 1
        return probs

    def _build_fidelity_graph(self, feats: torch.Tensor) -> nx.Graph:
        """
        Build a weighted graph from feature vectors using cosine similarity.
        """
        if feats.dim() == 3:
            feats = feats.mean(dim=0)
        n = feats.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        norms = torch.norm(feats, dim=1, keepdim=True) + 1e-12
        normalized = feats / norms
        sim = normalized @ normalized.t()
        for i in range(n):
            for j in range(i + 1, n):
                fid = sim[i, j].item()
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif self.secondary_threshold is not None and fid >= self.secondary_threshold:
                    graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    def _graph_propagate(self, feats: torch.Tensor, graph: nx.Graph) -> torch.Tensor:
        """
        Weighted aggregation of neighboring features.
        """
        n = feats.shape[0]
        out = feats.clone()
        for node in graph.nodes:
            neigh = list(graph.neighbors(node))
            if neigh:
                weights = torch.tensor(
                    [graph[node][nbr]["weight"] for nbr in neigh],
                    dtype=torch.float32,
                    device=feats.device,
                )
                neigh_feats = feats[neigh]
                weighted_sum = torch.sum(neigh_feats * weights[:, None], dim=0)
                out[node] += weighted_sum / weights.sum()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        """
        B = x.shape[0]
        patches = self._patch_features(x)  # [B, L]
        feats = patches.unsqueeze(-1)  # [B, L, 1]
        outputs = []
        for b in range(B):
            graph = self._build_fidelity_graph(feats[b].squeeze(-1))
            propagated = self._graph_propagate(feats[b].squeeze(-1), graph)
            outputs.append(propagated.mean(dim=0))
        return torch.stack(outputs)

__all__ = ["ConvGraphQNN"]
