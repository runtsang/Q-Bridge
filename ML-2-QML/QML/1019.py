import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionHybrid(nn.Module):
    """
    Hybrid quantum-classical network that replaces the classical 2x2 filter
    with a variational quantum kernel. For each 2x2 patch of the input image
    the patch values are encoded into a 4‑qubit circuit, a variational
    layer is applied, and the resulting measurement is mapped to an 8‑dim
    feature vector. All patch features are concatenated, flattened and
    fed to a linear classifier.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.num_patches = 14 * 14
        self.q_device = qml.device("default.qubit", wires=self.n_wires)
        # Variational circuit parameters: two layers with one parameter per qubit
        self.var_params = nn.Parameter(torch.randn(2, self.n_wires))
        # Mapping from measurement (4,) to feature vector (8,)
        self.mapping = nn.Linear(4, 8)
        # Classifier head
        self.linear = nn.Linear(self.num_patches * 8, 10)

    def _quantum_layer(self, data: torch.Tensor) -> torch.Tensor:
        """
        Quantum kernel applied to a single 2x2 patch.
        :param data: Tensor of shape (4,) with values in [0, 1].
        :return: Tensor of shape (4,) with measurement results.
        """
        @qml.qnode(self.q_device, interface="torch")
        def circuit(x):
            # Encode data into Ry rotations
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Variational layer
            for i in range(self.n_wires):
                qml.RY(self.var_params[0, i], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second variational layer
            for i in range(self.n_wires):
                qml.RY(self.var_params[1, i], wires=i)
            # Measure all qubits in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        return circuit(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        :param x: Tensor of shape (batch, 1, 28, 28)
        :return: Log‑softmax logits of shape (batch, 10)
        """
        batch_size = x.size(0)
        # Reshape to (batch, 28, 28)
        x = x.view(batch_size, 28, 28)
        patch_features = []
        for b in range(batch_size):
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = x[b, r:r+2, c:c+2].flatten()
                    # Normalize patch values to [0, 1]
                    patch = patch / 255.0
                    measurement = self._quantum_layer(patch)
                    feature = self.mapping(measurement)
                    patches.append(feature)
            # Concatenate all patch features for this sample
            sample_features = torch.cat(patches, dim=0)  # shape (num_patches, 8)
            patch_features.append(sample_features)
        # Stack batch
        patch_features = torch.stack(patch_features, dim=0)  # shape (batch, num_patches, 8)
        # Flatten
        flat = patch_features.view(batch_size, -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
