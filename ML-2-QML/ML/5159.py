import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from.quantum_module import QuanvolutionHybridModel as QuantumModule

class QuanvolutionHybridModel(nn.Module):
    """Classical + quantum hybrid model for MNIST‑style classification."""
    def __init__(self,
                 num_classes: int = 10,
                 conv_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 quantum_depth: int = 2,
                 quantum_shots: int = 1024):
        super().__init__()
        # Classical feature extractor
        self.classical_conv = nn.Conv2d(1, conv_channels,
                                        kernel_size=conv_kernel,
                                        stride=conv_stride)
        # Quantum kernel that processes 2×2 patches
        self.quantum_module = QuantumModule(num_qubits=conv_channels,
                                            depth=quantum_depth,
                                            shots=quantum_shots)
        # Linear head
        self.fc = nn.Linear(conv_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Classical feature extraction
        conv_out = self.classical_conv(x)  # (batch, conv_channels, 14, 14)
        # Reshape to a sequence of 2×2 patches encoded as qubits
        patches = conv_out.permute(0, 2, 3, 1).contiguous()  # (batch, 14, 14, conv_channels)
        patches = patches.view(batch_size, -1, conv_out.size(1))  # (batch, 196, conv_channels)
        # Forward through the quantum kernel
        patches_np = patches.detach().cpu().numpy()
        quantum_features = self.quantum_module.forward(patches_np)  # (batch, 196, conv_channels)
        quantum_features = torch.from_numpy(quantum_features).float()
        quantum_features = quantum_features.view(batch_size, -1)  # (batch, 196*conv_channels)
        # Linear classification head
        logits = self.fc(quantum_features)
        return F.log_softmax(logits, dim=-1)
