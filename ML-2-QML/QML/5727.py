import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumPatchEncoder(nn.Module):
    """
    Parameterized quantum circuit that encodes a 2×2 image patch (4 pixels)
    into a 4‑dimensional feature vector using a 4‑qubit circuit.
    """
    def __init__(self, num_params: int = 8):
        super().__init__()
        self.num_params = num_params
        self.weights = nn.Parameter(torch.randn(num_params))
        self.device = qml.device("default.qubit", wires=4)

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        # Encode pixel intensities into Ry rotations
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.RY(inputs[2], wires=2)
        qml.RY(inputs[3], wires=3)
        # Parameterized layer
        for i in range(self.num_params):
            qml.RY(weights[i], wires=i % 4)
            qml.CNOT(wires=[i % 4, (i + 1) % 4])
        # Return expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2)),
                qml.expval(qml.PauliZ(3))]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, 4) tensor of pixel values.
        Returns: (batch, 4) tensor of quantum features.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            out = qml.QNode(self._circuit, self.device,
                            interface="torch", diff_method="backprop")(
                inputs[i], self.weights)
            outputs.append(out)
        return torch.stack(outputs, dim=0)

class QuanvolutionGen112(nn.Module):
    """
    Hybrid quantum‑classical model that applies a quantum encoder to each
    2×2 patch, fuses the encoded patches with a classical convolutional
    layer, and classifies the resulting feature map.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 patch_size: int = 2, conv_out_channels: int = 16,
                 num_q_params: int = 8):
        super().__init__()
        self.patch_size = patch_size
        # Classical convolution to extract 2×2 patches
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size,
                                    stride=patch_size)
        # Quantum encoder for each patch
        self.quantum_encoder = QuantumPatchEncoder(num_params=num_q_params)
        # Classical convolution to fuse quantum features
        self.classical_conv = nn.Conv2d(4, conv_out_channels,
                                        kernel_size=3, padding=1)
        # Classifier head
        self.fc = nn.Linear(conv_out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        patches = self.patch_conv(x)  # (batch, 4, 14, 14)
        # Flatten spatial dimensions for quantum encoder
        patches_flat = patches.permute(0, 2, 3, 1).reshape(-1, 4)
        quantum_features = self.quantum_encoder(patches_flat)  # (batch*14*14, 4)
        # Reshape back to image‑like tensor
        quantum_features = quantum_features.view(patches.size(0), 14, 14, 4).permute(0, 3, 1, 2)
        fused = self.classical_conv(quantum_features)  # (batch, conv_out_channels, 14, 14)
        flattened = fused.view(fused.size(0), -1)
        logits = self.fc(flattened)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen112"]
