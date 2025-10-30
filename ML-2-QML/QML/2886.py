"""Quantum module implementing a quanvolutional filter and a graph‑based feature refinement.

The implementation uses Pennylane to simulate a 4‑qubit circuit on each 2×2 image patch and
Qutip to compute fidelities between the resulting state vectors. The class
`QuanvolutionGraphHybrid` mirrors the classical interface but executes the quantum
operations in a simulated backend.
"""

import pennylane as qml
import qutip as qt
import networkx as nx
import numpy as np

from typing import Tuple


class QuanvolutionGraphHybrid:
    """Quantum‑style quanvolutional network with graph‑based adjacency."""
    def __init__(self,
                 num_classes: int = 10,
                 patch_size: int = 2,
                 threshold: float = 0.9,
                 secondary: float | None = None):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.threshold = threshold
        self.secondary = secondary
        self.device = qml.device('default.qubit', wires=4)
        self.linear_weights = np.random.randn(4 * (28 // patch_size)**2, num_classes)
        self.linear_bias = np.random.randn(num_classes)

    def _quantum_patch(self, patch: np.ndarray) -> np.ndarray:
        @qml.qnode(self.device)
        def circuit(patch_vec):
            for i in range(4):
                qml.RY(patch_vec[i], wires=i)
            for i in range(4):
                qml.CNOT(wires=[i, (i+1) % 4])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        return circuit(patch)

    def _extract_patches(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        patches = []
        for r in range(0, h, self.patch_size):
            for c in range(0, w, self.patch_size):
                patch = img[r:r+self.patch_size, c:c+self.patch_size].flatten()
                patches.append(patch)
        return np.array(patches)

    def _state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.abs(np.vdot(a, b))**2

    def _graph_adjacency(self, features: np.ndarray) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(features.shape[0]))
        for i in range(features.shape[0]):
            for j in range(i+1, features.shape[0]):
                fid = self._state_fidelity(features[i], features[j])
                if fid >= self.threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif self.secondary is not None and fid >= self.secondary:
                    graph.add_edge(i, j, weight=self.secondary)
        return graph

    def forward(self, img: np.ndarray) -> Tuple[np.ndarray, nx.Graph]:
        patches = self._extract_patches(img)
        quantum_features = np.array([self._quantum_patch(p) for p in patches])
        flat_features = quantum_features.flatten()
        logits = flat_features @ self.linear_weights + self.linear_bias
        graph = self._graph_adjacency(quantum_features)
        return logits, graph
