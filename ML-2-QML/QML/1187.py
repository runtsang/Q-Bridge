import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumLayer(nn.Module):
    """
    Learnable quantum kernel that operates on 2×2 image patches.
    Uses a parameterised circuit with a small number of layers and
    measures all qubits in the Z basis.  The layer is fully differentiable
    and supports batched input via Pennylane's vectorised QNode.
    """

    def __init__(self,
                 num_qubits: int = 4,
                 num_layers: int = 3,
                 device: str = "default.qubit",
                 shots: int = 0):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits, shots=shots)
        # Learnable parameters: rotation angles for each layer and qubit
        self.params = nn.Parameter(torch.randn(num_layers, num_qubits, 3))
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch",
                               diff_method="backprop")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        # Input encoding: Ry gates with data
        for i in range(self.num_qubits):
            qml.RY(x[i], wires=i)

        # Parameterised layers
        for l in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RY(self.params[l, i, 0], wires=i)
                qml.RZ(self.params[l, i, 1], wires=i)
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measure all qubits in Pauli‑Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 4) where each row contains a 2×2 patch
        encoded as four pixel intensities.
        Returns: Tensor of shape (batch, 4) with the quantum feature vector.
        """
        return self.qnode(x)

class QuanvolutionNet(nn.Module):
    """
    Hybrid classical‑quantum network that applies a batched quantum
    filter to 2×2 patches of the input image, concatenates the
    resulting feature maps, and feeds them into a linear classifier.
    The quantum kernel is learnable and fully differentiable.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 q_device: str = "default.qubit",
                 shots: int = 0):
        super().__init__()
        self.quantum_layer = QuantumLayer(num_qubits=4,
                                          num_layers=3,
                                          device=q_device,
                                          shots=shots)
        # After 2×2 patches extraction we obtain 14×14 patches
        # Each patch yields a 4‑dimensional quantum feature vector
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2×2 patches from a batch of MNIST images.
        Returns a tensor of shape (batch, 14*14, 4).
        """
        bsz, _, h, w = x.shape
        assert h == 28 and w == 28, "Input must be 28×28 images."
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]
                # Flatten patch to 4 values
                patch = patch.view(bsz, -1)
                patches.append(patch)
        # Shape: (batch, 14*14, 4)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        patches = self._extract_patches(x)  # (batch, 196, 4)
        # Reshape to (batch*196, 4) for batched quantum evaluation
        batch_patches = patches.view(-1, 4)
        # Quantum feature extraction
        q_features = self.quantum_layer(batch_patches)  # (batch*196, 4)
        # Reshape back to (batch, 196, 4)
        q_features = q_features.view(x.size(0), 196, 4)
        # Flatten to (batch, 784)
        q_features = q_features.view(x.size(0), -1)
        # Classifier
        logits = self.fc(q_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
