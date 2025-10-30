"""Quantum implementation of the hybrid graph‑quanvolution network.

The quantum side mirrors the classical class above but replaces
the deterministic convolution with a two‑qubit quantum kernel
and the toy GNN with a variational layer that propagates a quantum
state through a sequence of parameterised unitaries.  The module
uses Qiskit for circuit construction and statevector simulation.

Key components
---------------
* ``QuantumQuanvolutionFilter`` – applies a random two‑qubit unitary
  to each 2×2 image patch and measures in the Z basis.
* ``QuantumGraphQNN`` – a variational network that evolves a state
  through layers of random unitaries and partial traces.
* ``QuantumGraphQuanvolutionHybrid`` – wrapper that forwards either
  an image tensor or a graph to the appropriate quantum routine.
"""

from __future__ import annotations

import itertools
import math
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.circuit.library import TwoLocal
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# 1. Quantum quanvolution layer
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(nn.Module):
    """Two‑qubit quantum kernel applied to every 2×2 patch.

    The filter is defined by a random unitary of size 4×4
    and a measurement in the computational basis.
    """

    def __init__(self, n_wires: int = 4, seed: int | None = None):
        super().__init__()
        self.n_wires = n_wires
        rng = np.random.default_rng(seed)
        unitary = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        self.unitary = np.linalg.qr(unitary)[0]  # orthonormalise
        self.circuit = QuantumCircuit(n_wires)
        self.circuit.unitary(self.unitary, range(n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unitary to each patch and return the flattened
        measurement outcomes as a real‑valued feature vector.
        """
        bsz = x.shape[0]
        device = x.device
        batch_features = []
        for img in x:
            img_np = img.cpu().numpy()
            img_np = img_np.reshape(28, 28)
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img_np[r:r+2, c:c+2].flatten()
                    # Encode the patch into |0…0> via rotations
                    qc = self.circuit.copy()
                    for idx, val in enumerate(patch):
                        qc.ry(val, idx)
                    qc.measure_all()
                    result = execute(qc, Aer.get_backend('qasm_simulator'),
                                     shots=1, memory=True).result()
                    measurement = np.fromstring(result.get_memory()[0], dtype=int)
                    patches.append(measurement.astype(np.float32))
            batch_features.append(np.concatenate(patches))
        return torch.tensor(batch_features, device=device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 2. Quantum graph‑based network
# --------------------------------------------------------------------------- #
class QuantumGraphQNN(nn.Module):
    """Variational network that propagates a quantum state through
    a sequence of random two‑qubit unitaries and performs partial traces.
    """

    def __init__(self, qnn_arch: list[int], seed: int | None = None):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.layers: list[QuantumCircuit] = []
        rng = np.random.default_rng(seed)
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            num_qubits = int(math.log2(in_f))
            qc = QuantumCircuit(num_qubits)
            # Random two‑qubit layer
            for _ in range(out_f):
                qc.append(TwoLocal(num_qubits, 'ry', 'cz', reps=1,
                                   param_init=rng.standard_normal, skip_reps=True),
                          range(num_qubits))
            self.layers.append(qc)

    def forward(self, state: Statevector) -> Statevector:
        """Propagate the input state through all layers."""
        for layer in self.layers:
            state = state.evolve(layer)
        return state

# --------------------------------------------------------------------------- #
# 3. Hybrid wrapper
# --------------------------------------------------------------------------- #
class QuantumGraphQuanvolutionHybrid(nn.Module):
    """Dispatches between quantum quanvolution and quantum graph‑based inference."""

    def __init__(self, qnn_arch: list[int], patch_size: int = 2, seed: int | None = None):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.quantum_quanvolution = QuantumQuanvolutionFilter(seed=seed)
        self.quantum_graph_qnn = QuantumGraphQNN(qnn_arch, seed=seed)
        self.classifier = nn.Linear((28 // patch_size) ** 2 * 4, 10)

    def forward(self, x: torch.Tensor | nx.Graph) -> torch.Tensor:
        if isinstance(x, nx.Graph):
            return self._forward_graph(x)
        elif isinstance(x, torch.Tensor):
            return self._forward_image(x)
        else:
            raise TypeError("Input must be torch.Tensor or networkx.Graph")

    def _forward_image(self, img: torch.Tensor) -> torch.Tensor:
        features = self.quantum_quanvolution(img)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def _forward_graph(self, G: nx.Graph) -> torch.Tensor:
        num_qubits = len(G.nodes())
        state = Statevector(random_statevector(2 ** num_qubits))
        state = self.quantum_graph_qnn(state)
        embedding = torch.tensor(state.data.real, dtype=torch.float32)
        logits = self.classifier(embedding.unsqueeze(0))
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuantumGraphQuanvolutionHybrid",
    "QuantumQuanvolutionFilter",
    "QuantumGraphQNN",
]
