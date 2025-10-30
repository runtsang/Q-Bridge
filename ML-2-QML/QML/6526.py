import torch
import torch.nn as nn
import pennylane as qml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuantumConvolutionFilter:
    """
    PennyLane quantum convolution filter that encodes 2×2 image patches
    into a 4‑qubit circuit and returns expectation‑value features.
    """
    def __init__(self, device: qml.Device, num_qubits: int = 4):
        self.device = device
        self.num_qubits = num_qubits

    def _patch_circuit(self, patch: torch.Tensor, params: torch.Tensor):
        @qml.qnode(self.device, interface="torch")
        def circuit(patch_tensor):
            # Encode patch values into rotations
            for i in range(self.num_qubits):
                qml.RY(patch_tensor[i], wires=i)
            # Entangle the qubits
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Parametric rotations controlled by circuit parameters
            for i in range(self.num_qubits):
                qml.RZ(params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return circuit(patch)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        batch, _, h, w = x.shape
        patch_size = 2
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = x[:, 0, i:i+patch_size, j:j+patch_size].reshape(batch, -1)
                # Pad to match number of qubits if needed
                if patch.shape[1] < self.num_qubits:
                    pad = torch.zeros(batch, self.num_qubits - patch.shape[1], device=patch.device)
                    patch = torch.cat([patch, pad], dim=1)
                # Apply quantum circuit for each patch in the batch
                patch_features = self._patch_circuit(patch, params)
                patches.append(patch_features)
        # Concatenate features from all patches
        features = torch.cat(patches, dim=1)
        return features

class FraudDetectionHybridModel(nn.Module):
    """
    Quantum hybrid fraud detection model.
    Uses a PennyLane quantum convolution filter followed by a variational classifier.
    """
    def __init__(
        self,
        conv_out_channels: int = 4,
        fraud_params: Optional[List[FraudLayerParameters]] = None,
    ):
        super().__init__()
        # Quantum device with 4 qubits for the convolution filter and classifier
        self.device = qml.device("default.qubit", wires=4, shots=1024)
        self.conv_filter = QuantumConvolutionFilter(self.device)
        # Parameters for the patch circuit
        self.patch_params = nn.Parameter(torch.randn(4))
        # Classifier parameters: one weight per feature plus a bias
        num_features = conv_out_channels * 14 * 14
        self.classifier_params = nn.Parameter(torch.randn(num_features + 1))
        self.fraud_params = fraud_params

    def _classifier_circuit(self, features: torch.Tensor):
        @qml.qnode(self.device, interface="torch")
        def circuit(feat_tensor):
            for i, f in enumerate(feat_tensor):
                qml.RY(f * self.classifier_params[i], wires=0)
            qml.RZ(self.classifier_params[-1], wires=0)
            return qml.expval(qml.PauliZ(0))
        return circuit(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        features = self.conv_filter.forward(x, self.patch_params)
        # Flatten features for the classifier
        features_flat = features.reshape(x.shape[0], -1)
        probs = self._classifier_circuit(features_flat)
        return torch.sigmoid(probs)

__all__ = ["FraudLayerParameters", "FraudDetectionHybridModel"]
