"""Hybrid binary classifier with quantum backbone.

The implementation mirrors the classical version but replaces the
hybrid head and transformer block with quantum modules.  The
quantum head is a single‑qubit parameterised circuit; the
transformer block is a quantum‑enhanced attention/FFN
as defined in QTransformerTorch.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Tuple

# ----------------------------------------------------------------------
# Data utilities – same as in the classical module
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic binary data derived from a sinusoidal function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y = (y > 0).astype(np.float32)
    return x, y

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Quantum circuit – adapted from FCL.py
# ----------------------------------------------------------------------
class QuantumCircuit:
    """One‑qubit parameterised circuit used as a fully‑connected layer."""
    def __init__(self, backend, shots: int = 100):
        self._circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(0)
        self._circuit.ry(self.theta, 0)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        return np.array([expectation(result)])

# ----------------------------------------------------------------------
# Hybrid activation – connects PyTorch to the quantum circuit
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards a scalar to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = 0.0) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(thetas + shift)
        result = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy().flatten()
        gradients = []
        for theta in thetas:
            exp_plus = ctx.circuit.run([theta + shift])
            exp_minus = ctx.circuit.run([theta - shift])
            gradients.append(exp_plus - exp_minus)
        grad = torch.tensor(gradients, device=inputs.device, dtype=inputs.dtype)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head that replaces the classical sigmoid."""
    def __init__(self, backend, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# ----------------------------------------------------------------------
# CNN backbone – identical to the classical version
# ----------------------------------------------------------------------
class CNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------------------------------------------------
# Quantum transformer block – from QTransformerTorch.py
# ----------------------------------------------------------------------
class QAttentionLayer(tq.QuantumModule):
    """Quantum multi‑head attention realised with a single‑qubit encoder."""
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate in self.parameters:
            gate(qdev)
        return self.measure(qdev)

class QFeedForward(tq.QuantumModule):
    """Quantum feed‑forward layer."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate in self.parameters:
            gate(qdev)
        out = self.measure(qdev)
        out = self.linear1(out)
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum‑enhanced transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires: int = 8):
        super().__init__()
        self.attn = QAttentionLayer(embed_dim, num_heads, n_wires)
        self.ffn = QFeedForward(embed_dim, ffn_dim, n_wires)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        qdev = qdev or tq.QuantumDevice(n_wires=self.attn.n_wires, bsz=x.size(0), device=x.device)
        attn_out = self.attn(x, qdev)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, qdev)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerEncoderQuantum(tq.QuantumModule):
    """Stack of quantum transformer blocks."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int, n_wires: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_wires) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, qdev)
        return x

# ----------------------------------------------------------------------
# Full hybrid quantum classifier
# ----------------------------------------------------------------------
class HybridBinaryClassifier(nn.Module):
    """CNN + quantum transformer + quantum head for binary classification."""
    def __init__(self,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 num_blocks: int = 2,
                 backend: qiskit.providers.basebackend.BaseBackend = qiskit.Aer.get_backend("aer_simulator"),
                 shots: int = 200,
                 shift: float = np.pi / 2):
        super().__init__()
        self.backbone = CNNBackbone()
        self.transformer = TransformerEncoderQuantum(1, num_heads, ffn_dim, num_blocks)
        self.hybrid = Hybrid(backend, shots, shift)
        self.proj = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.backbone(x)
        seq = x.unsqueeze(1)
        qdev = None  # let the transformer create its own device
        seq = self.transformer(seq, qdev)
        logits = self.hybrid(seq.squeeze(1))
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return probs

__all__ = [
    "HybridBinaryClassifier",
    "ClassificationDataset",
    "generate_superposition_data",
    "HybridFunction",
    "Hybrid",
    "QuantumCircuit",
]
