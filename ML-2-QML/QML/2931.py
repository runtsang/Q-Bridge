"""Quantum‑centric quanvolutional model combining a quantum filter and a quantum self‑attention block.

The module defines :class:`QuanvolutionHybrid` that mirrors the classical
implementation but replaces the feature extractor and attention with quantum
circuits executed on a Qiskit simulator.  The quantum filter encodes each
2×2 image patch into a 4‑qubit circuit and measures the outcome, while the
quantum self‑attention block computes a weight for each patch based on a
controlled‑rotation circuit.  The weighted features are then passed through a
classical linear classifier.

This design allows direct comparison of quantum‑enhanced feature extraction
and attention against a purely classical baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, Aer, execute

__all__ = ["QuanvolutionHybrid"]


class QuantumQuanvolutionFilter(nn.Module):
    """Quantum filter that encodes 2×2 patches into a 4‑qubit circuit.

    The filter splits the input image into non‑overlapping 2×2 patches, encodes
    each patch into a 4‑qubit circuit via Ry rotations, applies a short random
    entanglement layer, and measures all qubits.  The measurement
    probabilities are returned as a feature vector for each patch.
    """

    def __init__(self, n_qubits: int = 4, patch_size: int = 2, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _encode_patch(self, patch: np.ndarray) -> np.ndarray:
        # Flatten patch to a vector of length patch_size*patch_size
        flat = patch.flatten()
        # Pad or truncate to match number of qubits
        if len(flat) < self.n_qubits:
            flat = np.pad(flat, (0, self.n_qubits - len(flat)), "constant")
        else:
            flat = flat[: self.n_qubits]
        # Build circuit
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(flat):
            qc.ry(val, i)
        # Random entanglement layer
        for _ in range(3):
            i = np.random.randint(self.n_qubits - 1)
            qc.cx(i, i + 1)
        qc.measure_all()
        # Execute
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        probs = np.zeros(2**self.n_qubits)
        for bitstring, cnt in result.items():
            idx = int(bitstring, 2)
            probs[idx] = cnt / self.shots
        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
           .unfold(3, self.patch_size, self.patch_size)
           .contiguous()
        )  # (B, C, n_h, n_w, patch_size, patch_size)
        n_patches = patches.shape[2] * patches.shape[3]
        features = []
        for b in range(B):
            patch_feats = []
            for i in range(patches.shape[2]):
                for j in range(patches.shape[3]):
                    patch = patches[b, :, i, j].cpu().numpy()
                    probs = self._encode_patch(patch)
                    patch_feats.append(probs)
            features.append(np.stack(patch_feats))
        return torch.tensor(np.stack(features), dtype=torch.float32)  # (B, N, 2**n_qubits)


class QuantumSelfAttention(nn.Module):
    """Quantum self‑attention that assigns a weight to each patch.

    For each patch feature vector, a 4‑qubit circuit encodes the first four
    components via Ry rotations, applies controlled‑RX entanglement, measures,
    and the probability of measuring a |1⟩ on the first qubit is used as the
    attention weight.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _weight_from_patch(self, patch_feat: np.ndarray) -> float:
        # Use first n_qubits components (or pad)
        if len(patch_feat) < self.n_qubits:
            patch_feat = np.pad(patch_feat, (0, self.n_qubits - len(patch_feat)), "constant")
        else:
            patch_feat = patch_feat[: self.n_qubits]
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(patch_feat):
            qc.ry(val, i)
        # Entanglement: controlled‑RX
        for i in range(self.n_qubits - 1):
            qc.crx(np.pi / 4, i, i + 1)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        # Weight = probability of measuring a |1⟩ on the first qubit
        weight = sum(cnt for bit, cnt in result.items() if bit[-1] == "1") / self.shots
        return weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, N, dim)
        B, N, _ = features.shape
        weights = []
        for b in range(B):
            patch_weights = []
            for n in range(N):
                patch_feat = features[b, n].detach().cpu().numpy()
                w = self._weight_from_patch(patch_feat)
                patch_weights.append(w)
            weights.append(patch_weights)
        return torch.tensor(np.array(weights), dtype=torch.float32)  # (B, N)


class QuanvolutionHybrid(nn.Module):
    """Hybrid quanvolutional classifier with quantum filter and quantum self‑attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits per patch (default 4).
    num_classes : int
        Number of target classes (default 10 for MNIST).
    """

    def __init__(self, n_qubits: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter(n_qubits=n_qubits)
        self.attention = QuantumSelfAttention(n_qubits=n_qubits)
        self.classifier = nn.Linear(2**n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        features = self.filter(x)  # (B, N, 2**n_qubits)
        weights = self.attention(features)  # (B, N)
        weighted = features * weights.unsqueeze(-1)
        flat = weighted.view(x.size(0), -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)
