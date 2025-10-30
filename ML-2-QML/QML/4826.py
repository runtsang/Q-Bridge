from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

class ConvQNN(nn.Module):
    """Quantum‑augmented convolutional network with graph‑based fidelity adjacency."""
    def __init__(
        self,
        kernel_size: int = 2,
        n_layers: int = 1,
        threshold: float = 0.8,
        secondary: float | None = 0.6,
        secondary_weight: float = 0.5,
        shots: int = 200,
        backend_name: str = "qasm_simulator",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight
        self.shots = shots
        self.backend = Aer.get_backend(backend_name)

        # Parameter‑shared RandomLayer circuit
        self.circuit = self._build_circuit()
        self.head = None

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        """Build a shared RandomLayer circuit with depth n_layers."""
        n_wires = self.kernel_size ** 2
        circuit = qiskit.QuantumCircuit(n_wires)
        params = ParameterVector("theta", length=n_wires * self.n_layers)
        for i, param in enumerate(params):
            circuit.rx(param, i % n_wires)
        circuit += random_circuit(n_wires, 2)
        circuit.measure_all()
        return circuit

    def _encode_patch(self, patch: np.ndarray) -> qiskit.circuit.ParameterBinding:
        """Bind pixel values to rotation angles."""
        bind = {}
        for i, val in enumerate(patch):
            angle = math.pi if val > 0.5 else 0.0
            bind[self.circuit.params[i]] = angle
        return bind

    def _measure(self, bind: qiskit.circuit.ParameterBinding) -> np.ndarray:
        """Run circuit and return per‑qubit probability of measuring |1>."""
        bound_circ = self.circuit.bind_parameters(bind)
        state = Statevector(bound_circ)
        probs = np.zeros(self.kernel_size ** 2)
        for idx, amplitude in enumerate(state.data):
            prob = abs(amplitude) ** 2
            bits = format(idx, f"0{self.kernel_size**2}b")
            for q, bit in enumerate(bits):
                if bit == "1":
                    probs[q] += prob
        return probs

    def _build_adjacency(self, probs: np.ndarray) -> np.ndarray:
        """Adjacency from cosine similarity of per‑qubit probabilities."""
        norms = probs / np.linalg.norm(probs, axis=1, keepdims=True)
        sim = norms @ norms.T
        adj = np.zeros_like(sim)
        adj[sim >= self.threshold] = 1
        if self.secondary is not None:
            mask = (sim >= self.secondary) & (sim < self.threshold)
            adj[mask] = self.secondary_weight
        return adj

    def _propagate(self, probs: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Graph Laplacian propagation."""
        deg = adj.sum(axis=1, keepdims=True)
        lap = deg - adj
        return lap @ probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (B, 10)."""
        bsz, _, H, W = x.shape
        patch_size = self.kernel_size
        npatch_h = (H - patch_size) // patch_size + 1
        npatch_w = (W - patch_size) // patch_size + 1
        logits_list = []

        for b in range(bsz):
            probs_img = []
            for i in range(npatch_h):
                for j in range(npatch_w):
                    patch = x[b, 0,
                              i*patch_size:(i+1)*patch_size,
                              j*patch_size:(j+1)*patch_size].detach().cpu().numpy()
                    bind = self._encode_patch(patch.flatten())
                    prob = self._measure(bind)
                    probs_img.append(prob)
            probs_img = np.stack(probs_img, axis=0)  # (n_patches, n_qubits)
            adj = self._build_adjacency(probs_img)
            propagated = self._propagate(probs_img, adj)
            flat = propagated.reshape(-1)
            if self.head is None:
                self.head = nn.Linear(flat.shape[0], 10)
            logits = self.head(torch.from_numpy(flat).float())
            logits_list.append(logits)
        return torch.stack(logits_list, dim=0)
