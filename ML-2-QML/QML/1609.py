import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionNet(nn.Module):
    """
    Quantum‑augmented hybrid model that uses a variational circuit to encode
    each 2×2 patch into a 4‑qubit state and measures Pauli‑Z expectations.
    The resulting quantum feature map is concatenated with a classical
    4×4 patch extractor for richer representations.
    """
    def __init__(self, n_classes: int = 10, n_layers: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        # Parameters for the variational layers
        self.params = nn.Parameter(torch.randn(n_layers, 4))
        # Quantum device
        self.dev = qml.device("default.qubit", wires=4)

        # Define a QNode that applies the variational circuit to a batch of patches
        def _circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            """
            Apply the variational circuit to each patch in a batch.

            Parameters
            ----------
            inputs : torch.Tensor
                Batch of 2×2 patch vectors of shape (batch, 4).
            params : torch.Tensor
                Variational parameters of shape (n_layers, 4).

            Returns
            -------
            torch.Tensor
                Expectation values of shape (batch, 4).
            """
            def inner(v: torch.Tensor) -> torch.Tensor:
                # Encode classical data via Ry rotations
                for i, val in enumerate(v):
                    qml.RY(val, wires=i)
                # Variational layers
                for layer in range(self.n_layers):
                    theta = params[layer]
                    qml.RY(theta[0], wires=0)
                    qml.RY(theta[1], wires=1)
                    qml.CNOT(wires=[0, 1])
                    qml.RY(theta[2], wires=2)
                    qml.RY(theta[3], wires=3)
                    qml.CNOT(wires=[2, 3])
                # Measure Pauli‑Z on all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(4)]
            return qml.map(inner, inputs)

        self.qnode = qml.QNode(_circuit, self.dev, interface="torch")

        # Classical patch extractor for 4×4 patches
        self.unfold_4x4 = nn.Unfold(kernel_size=4, stride=4)
        # Compute output feature dimension
        feat_4x4 = 4 * 7 * 7  # 7×7 patches from 4×4 kernel
        self.linear = nn.Linear(4 * 14 * 14 + feat_4x4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, n_classes).
        """
        # Extract 2×2 patches and encode them quantumly
        patches_2x2 = nn.Unfold(kernel_size=2, stride=2)(x)  # (batch, 4, 196)
        batch_size, _, n_patches = patches_2x2.shape
        quantum_features = []
        for i in range(n_patches):
            # Each patch vector: (batch, 4)
            patch_vec = patches_2x2[:, :, i]
            out = self.qnode(patch_vec, self.params)  # (batch, 4)
            quantum_features.append(out)
        quantum_features = torch.cat(quantum_features, dim=1)  # (batch, 4*196)

        # Classical 4×4 patch extractor
        patches_4x4 = self.unfold_4x4(x)  # (batch, 4, 49)
        classical_features = patches_4x4.view(x.size(0), -1)  # (batch, 4*49)

        # Concatenate quantum and classical features
        features = torch.cat([quantum_features, classical_features], dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
