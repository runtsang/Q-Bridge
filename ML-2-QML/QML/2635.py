import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumPatchExtractor(nn.Module):
    """Quantum patch extractor that processes 2×2 image patches via a 4‑qubit circuit."""
    def __init__(self, n_qubits: int = 4, n_ops: int = 8, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_ops = n_ops
        self.shots = shots
        self.backend = AerSimulator()
        # Build a reusable circuit template
        self.circuit_template = QuantumCircuit(n_qubits)
        # Encode pixel values as Ry rotations
        for i in range(n_qubits):
            self.circuit_template.ry(f"θ_{i}", i)
        # Add a random two‑qubit layer
        for _ in range(n_ops):
            a, b = torch.randint(0, n_qubits, (2,)).tolist()
            self.circuit_template.cx(a, b)
        self.circuit_template.measure_all()

    def _expectation(self, counts: dict) -> float:
        exp = 0.0
        for state, cnt in counts.items():
            # state is a bitstring like '0101'
            z = 1 if state.count('1') % 2 == 0 else -1
            exp += z * cnt / self.shots
        return exp

    def run_patch(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (4,) tensor of pixel values
        theta_vals = {f"θ_{i}": val.item() for i, val in enumerate(patch)}
        circ = self.circuit_template.copy()
        circ = transpile(circ, self.backend)
        qobj = assemble(circ, shots=self.shots, parameter_binds=[theta_vals])
        result = self.backend.run(qobj).result()
        counts = result.get_counts(circ)
        exp = self._expectation(counts)
        return torch.tensor(exp, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ps = 2
        assert H % ps == 0 and W % ps == 0, "Image dimensions must be divisible by patch size"
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(B, C, -1, ps, ps)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, -1, C * ps * ps)
        feats = []
        for b in range(B):
            batch_feats = []
            for patch in patches[b]:
                flat = patch[:4]  # take first 4 values for grayscale
                flat = flat / 255.0
                exp = self.run_patch(flat)
                batch_feats.append(exp)
            feats.append(torch.stack(batch_feats))
        feats = torch.stack(feats)  # (B, num_patches)
        return feats

class QuantumHybridHead(nn.Module):
    """Quantum hybrid head that maps a feature vector to a probability via a parameterised circuit."""
    def __init__(self, in_features: int, n_qubits: int = 1, shots: int = 1024):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit_template = QuantumCircuit(n_qubits)
        self.circuit_template.ry("θ", 0)
        self.circuit_template.measure_all()

    def _expectation(self, counts: dict) -> float:
        exp = 0.0
        for state, cnt in counts.items():
            z = 1 if state == '0' else -1
            exp += z * cnt / self.shots
        return exp

    def run_head(self, features: torch.Tensor) -> torch.Tensor:
        # features: (in_features,)
        theta = features.mean().item()
        circ = self.circuit_template.copy()
        circ = transpile(circ, self.backend)
        qobj = assemble(circ, shots=self.shots, parameter_binds=[{"θ": theta}])
        result = self.backend.run(qobj).result()
        counts = result.get_counts(circ)
        exp = self._expectation(counts)
        return torch.tensor(exp, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for batch in x:
            exp = self.run_head(batch)
            probs.append(exp)
        probs = torch.stack(probs)
        return torch.sigmoid(probs)

class UnifiedQuanvolutionHybrid(nn.Module):
    """Hybrid model that uses a quantum patch extractor and a quantum hybrid head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.patch_extractor = QuantumPatchExtractor()
        # For 28×28 images: 14×14 patches → 196 patches
        self.classifier = nn.Linear(196, num_classes)
        self.hybrid_head = QuantumHybridHead(in_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(x)
        logits = self.classifier(patches)
        probs = self.hybrid_head(logits)
        return F.log_softmax(probs, dim=-1)

__all__ = ["UnifiedQuanvolutionHybrid"]
