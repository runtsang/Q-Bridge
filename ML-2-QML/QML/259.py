"""Quantum quanvolution filter with multi‑scale feature extraction using Pennylane.

The quantum branch encodes 2×2 image patches into 4 qubits, applies a trainable variational circuit, and measures Pauli‑Z expectation values.  Classical depth‑wise separable convolutions handle 4×4 and 8×8 patches, and the outputs are concatenated with a residual shortcut before flattening and classification.  The module is fully compatible with torch autograd and can be trained end‑to‑end on CPU, GPU, or a quantum simulator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionGen287(nn.Module):
    """Quantum quanvolution with multi‑scale feature extraction."""
    def __init__(self, in_channels: int = 1, out_classes: int = 10,
                 n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # Variational parameters
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        # Classical 4×4 and 8×8 depth‑wise separable convs
        self.dw4 = nn.Conv2d(in_channels, 4, kernel_size=4, stride=4, groups=in_channels)
        self.pw4 = nn.Conv2d(4, 4, kernel_size=1)
        self.dw8 = nn.Conv2d(in_channels, 4, kernel_size=8, stride=8, groups=in_channels)
        self.pw8 = nn.Conv2d(4, 4, kernel_size=1)
        # Residual shortcut
        self.residual = nn.Conv2d(in_channels, 4, kernel_size=1)
        # Linear head
        self.head = nn.Linear(4 * 3 * 14 * 14, out_classes)

        # Quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            # Encode inputs into qubits with Ry rotations
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                # Entanglement
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def _quantum_feature(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute quantum features for 2×2 patches.

        Args:
            patches: Tensor of shape (batch, 14, 14, 4) with pixel values.
        Returns:
            Tensor of shape (batch, 14, 14, 4) with expectation values.
        """
        b, h, w, c = patches.shape
        flat = patches.reshape(-1, 4)
        qfeat = torch.stack([self._circuit(p, self.params) for p in flat])
        return qfeat.reshape(b, h, w, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # Residual
        res = self.residual(x)
        # 2×2 patches
        patches2 = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous()
        patches2 = patches2.view(bsz, -1, 4)
        # Reshape to (batch, 14, 14, 4)
        patches2 = patches2.view(bsz, 14, 14, 4)
        feat2 = self._quantum_feature(patches2)
        # 4×4 classical features
        patches4 = x.unfold(2, 4, 4).unfold(3, 4, 4).contiguous()
        patches4 = patches4.view(bsz, -1, 4)
        feat4 = F.relu(self.pw4(self.dw4(x)))
        # 8×8 classical features
        patches8 = x.unfold(2, 8, 8).unfold(3, 8, 8).contiguous()
        patches8 = patches8.view(bsz, -1, 4)
        feat8 = F.relu(self.pw8(self.dw8(x)))
        # Concatenate quantum and classical features
        out = torch.cat([feat2, feat4, feat8], dim=3) + res
        # Flatten
        flat = out.view(bsz, -1)
        logits = self.head(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen287"]
