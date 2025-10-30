import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """Quantum filter that applies a parameterized variational circuit to each
    2×2 image patch. The circuit encodes pixel values into rotation angles
    and shares a trainable set of parameters across all patches."""
    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 device_name: str = "default.qubit"):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Shared trainable parameters for the variational layers
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))
        # Quantum device
        self.dev = qml.device(device_name, wires=n_wires)

    def _quantum_circuit(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply the variational circuit to a single 2×2 patch.

        Args:
            patch: Tensor of shape (4,) containing pixel values.
        Returns:
            Tensor of shape (4,) with expectation values of Pauli‑Z.
        """
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(patch):
            # Encode pixel values as rotations
            for i in range(self.n_wires):
                qml.RY(patch[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_wires):
                    qml.RY(self.params[layer, i], wires=i)
                # Entangling pattern (cyclic CNOTs)
                for i in range(self.n_wires):
                    qml.CNOT(wires=[i, (i + 1) % self.n_wires])
            # Measure expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        return circuit(patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        x = x.view(B, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch and flatten
                patch = torch.stack([x[:, r, c],
                                     x[:, r, c + 1],
                                     x[:, r + 1, c],
                                     x[:, r + 1, c + 1]], dim=1)  # (B, 4)
                # Apply quantum circuit to each sample in the batch
                patch_features = torch.stack([self._quantum_circuit(patch[i])
                                              for i in range(B)], dim=0)  # (B, 4)
                patches.append(patch_features)
        # Concatenate all patch features: (B, 4 * 196) = (B, 784)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quantum filter followed by a linear
    classification head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
