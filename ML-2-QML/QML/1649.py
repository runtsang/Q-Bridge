import torch
import torch.nn as nn
import pennylane as qml

class QuanvolutionGen224(nn.Module):
    """
    Hybrid quantum–classical auto‑encoder that uses a 4‑qubit variational
    circuit to extract 4‑dimensional quantum features from image
    patches, compresses them to 224 latent dimensions, and then
    reconstructs the original 28×28 image via a classical decoder.
    """
    def __init__(self) -> None:
        super().__init__()
        # Quantum device and circuit parameters
        self.device = qml.device("default.qubit", wires=4)
        self.var_params = nn.Parameter(torch.randn(4, requires_grad=True))
        self.qnode = qml.QNode(self._variational_circuit,
                               self.device,
                               interface="torch",
                               diff_method="backprop")

        # Classical mapping to 224‑dim latent space
        self.latent = nn.Linear(4, 224)

        # Decoder mirroring the classical auto‑encoder
        self.dec_fc = nn.Linear(224, 8 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (8, 7, 7))
        self.deconv1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)

    def _variational_circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Simple 4‑qubit variational ansatz with parameterised RZ gates
        followed by a ring of CNOTs.
        """
        for i in range(4):
            qml.RY(x[i], wires=i)
        for i in range(4):
            qml.RZ(params[i], wires=i)
            qml.CNOT(i, (i + 1) % 4)
        return torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image batch of shape (batch, 1, 28, 28).
        Returns:
            Reconstructed images of shape (batch, 1, 28, 28).
        """
        bsz = x.shape[0]
        # Extract four coarse‑grained features from the image
        quadrants = torch.stack([
            x[:, :, :14, :14].mean(dim=[2, 3]),
            x[:, :, :14, 14:].mean(dim=[2, 3]),
            x[:, :, 14:, :14].mean(dim=[2, 3]),
            x[:, :, 14:, 14:].mean(dim=[2, 3])
        ], dim=1)  # (bsz, 4)

        # Quantum feature extraction (batch‑wise via loop)
        q_features = torch.stack(
            [self.qnode(quadrants[i], self.var_params) for i in range(bsz)],
            dim=0
        )  # (bsz, 4)

        latent = torch.relu(self.latent(q_features))
        x = torch.relu(self.dec_fc(latent))
        x = self.unflatten(x)
        x = torch.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x
