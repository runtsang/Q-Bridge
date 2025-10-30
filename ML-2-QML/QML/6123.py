import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
from typing import List, Tuple

class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum‑centric implementation of the hybrid model.  The forward method receives a
    classical tensor, constructs a fidelity‑based graph, and applies a per‑edge
    parameterized unitary.  The model is fully differentiable and can be
    trained with a classical optimiser via the torchquantum backend.
    """

    class QGraphLayer(tq.QuantumModule):
        """
        Quantum layer that applies a random circuit followed by a trainable
        two‑qubit gate for each edge in the adjacency graph.
        """
        def __init__(self, n_wires: int, edge_list: List[Tuple[int, int]]):
            super().__init__()
            self.n_wires = n_wires
            self.edge_list = edge_list
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Trainable parameters for each edge (U3 gate: 3 rotation angles + phase)
            self.edge_params = nn.ParameterList(
                [nn.Parameter(torch.randn(4)) for _ in edge_list]
            )

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for (i, j), params in zip(self.edge_list, self.edge_params):
                tqf.u3(qdev, params, wires=[i, j], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, in_channels: int = 1, num_classes: int = 4, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.quantum_layer = None  # instantiated in forward
        self.head = nn.Sequential(
            nn.Linear(n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def _build_fidelity_graph(self, features: torch.Tensor) -> nx.Graph:
        norms = features.norm(dim=1, keepdim=True)
        sims = (features @ features.t()) / (norms @ norms.t() + 1e-12)
        g = nx.Graph()
        g.add_nodes_from(range(features.size(0)))
        threshold = 0.9
        for i in range(features.size(0)):
            for j in range(i + 1, features.size(0)):
                if sims[i, j] >= threshold:
                    g.add_edge(i, j, weight=1.0)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Classical encoder
        feat = self.encoder(x)
        feat_flat = feat.view(batch_size, -1)
        # Build graph over batch
        graph = self._build_fidelity_graph(feat_flat)
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device, record_op=True)
        # Encode features into qubits
        self.encoder(qdev, feat_flat)
        # Instantiate quantum graph layer with current graph edges
        edge_list = list(graph.edges)
        self.quantum_layer = self.QGraphLayer(self.n_wires, edge_list)
        self.quantum_layer(qdev)
        # Measurement
        out = tq.MeasureAll(tq.PauliZ)(qdev)
        # Classifier
        logits = self.head(out)
        return self.norm(logits)
