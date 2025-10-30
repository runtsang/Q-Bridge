"""Hybrid quantum‑classical network that mirrors the classical model.

The quantum module uses a variational circuit (RealAmplitudes) to process
classical image patches, a sampler to obtain expectation values,
and an interpret function that builds a fidelity graph on the
measurement outcomes.  The final logits are produced by a small
classical linear head.  This design demonstrates how quantum
circuits can provide non‑linear feature maps that are then
structured by graph‑based clustering before classification.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Patch extraction helper ----------------------------------------------- #
# --------------------------------------------------------------------------- #
def extract_patches(img: np.ndarray, patch_size: int = 2) -> np.ndarray:
    """Return a 2‑D array of flattened patches."""
    h, w = img.shape
    patches = []
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            patch = img[r:r+patch_size, c:c+patch_size].flatten()
            patches.append(patch)
    return np.stack(patches, axis=0)

# --------------------------------------------------------------------------- #
# 2. Variational circuit ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def create_variational_circuit(num_qubits: int, reps: int = 3):
    """RealAmplitudes circuit that serves as a quantum kernel."""
    return RealAmplitudes(num_qubits, reps=reps)

# --------------------------------------------------------------------------- #
# 3. Fidelity‑based graph ----------------------------------------------- #
# --------------------------------------------------------------------------- #
def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Absolute squared overlap between two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2

def build_fidelity_graph(states: list[Statevector], threshold: float,
                         secondary: float | None = None,
                         secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j, aj in enumerate(states[i+1:], start=i+1):
            fid = state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G

def cluster_centroids(states: list[Statevector], G: nx.Graph) -> np.ndarray:
    """Return centroid amplitudes per connected component."""
    centroids = []
    for comp in nx.connected_components(G):
        comp_vecs = [states[i].data for i in comp]
        centroids.append(np.mean(comp_vecs, axis=0))
    return np.vstack(centroids)

# --------------------------------------------------------------------------- #
# 4. Interpret function ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def interpret_fidelity(measurements: np.ndarray,
                       threshold: float = 0.7,
                       secondary: float | None = None) -> np.ndarray:
    """Map measurement probabilities to cluster logits."""
    # Convert measurement probabilities to Statevectors
    states = [Statevector(prob) for prob in measurements]
    G = build_fidelity_graph(states, threshold, secondary)
    centroids = cluster_centroids(states, G)
    return centroids

# --------------------------------------------------------------------------- #
# 5. Hybrid QNN ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
class HybridQuanvolutionQNN:
    """Quantum neural network that mimics the classical hybrid architecture."""
    def __init__(self, num_qubits: int = 4, num_classes: int = 10,
                 fidelity_thr: float = 0.7, secondary_thr: float | None = 0.5):
        self.num_qubits = num_qubits
        self.fidelity_thr = fidelity_thr
        self.secondary_thr = secondary_thr
        self.circuit = create_variational_circuit(num_qubits)
        self.sampler = qiskit.primitives.StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda m: interpret_fidelity(
                m, threshold=self.fidelity_thr, secondary=self.secondary_thr
            ),
            output_shape=(num_qubits,),
            sampler=self.sampler,
        )
        self.classifier = qiskit.quantum_info.Statevector(np.eye(num_classes))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Run the hybrid network on a single image."""
        patches = extract_patches(img)
        # Feed each patch to the sampler
        probs = self.qnn(patches)
        # The interpret function already returns cluster centroids
        logits = probs @ self.classifier.data
        return logits

__all__ = ["HybridQuanvolutionQNN"]
