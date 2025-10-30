import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen(nn.Module):
    """
    Quantum‑classical hybrid model using PennyLane.

    The filter applies a parameterized circuit to 2×2 image patches.
    Each patch is encoded with Ry rotations, followed by a trainable
    entangling block.  The circuit returns both the expectation of
    PauliZ (amplitude) and PauliY (phase).  The resulting feature map
    is flattened and passed to a classical linear head.
    """
    def __init__(self, in_channels: int = 1, out_classes: int = 10, use_quantum_head: bool = False):
        super().__init__()
        self.patch_size = 2
        self.n_wires = 4
        self.use_quantum_head = use_quantum_head

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        # Trainable parameters for the entangling layer
        self.entanglement_weights = nn.Parameter(torch.randn(4))

        # Classical head
        self.classical_head = nn.Linear(8 * 14 * 14, out_classes)

        # Quantum head
        if use_quantum_head:
            self.q_head_weights = nn.Parameter(torch.randn(out_classes, 8 * 14 * 14))

    def _quantum_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Return amplitude and phase for a single 2×2 patch."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Encode pixel values with Ry
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Trainable entangling block
            for i in range(self.n_wires):
                qml.CNOT(wires=[i, (i + 1) % self.n_wires])
                qml.RY(self.entanglement_weights[i], wires=i)
                qml.CNOT(wires=[i, (i + 1) % self.n_wires])
            # Return amplitude (PauliZ) and phase (PauliY) of qubit 0
            amp = qml.expval(qml.PauliZ(wires=0))
            phase = qml.expval(qml.PauliY(wires=0))
            return amp, phase

        return circuit(patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                amp, phase = self._quantum_patch(patch)
                patch_feat = torch.cat([amp, phase], dim=1)
                patches.append(patch_feat)

        features = torch.cat(patches, dim=1)

        if self.use_quantum_head:
            # Quantum linear head: each class is a qubit
            logits = torch.zeros(bsz, self.classical_head.out_features, device=x.device)
            for i in range(self.classical_head.out_features):
                @qml.qnode(self.dev, interface="torch")
                def qhead(x):
                    for j, val in enumerate(x):
                        qml.RY(val, wires=j)
                    qml.RY(self.q_head_weights[i, 0], wires=0)
                    return qml.expval(qml.PauliZ(wires=0))
                logits[:, i] = qhead(features[:, i])
        else:
            logits = self.classical_head(features)

        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen"]
