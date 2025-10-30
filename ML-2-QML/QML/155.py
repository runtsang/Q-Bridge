import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionHybrid(nn.Module):
    """
    Quantum‑classical hybrid model that replaces the random quantum layer
    of the original Quanvolution with a trainable variational ansatz.
    Each 2×2 patch is encoded into a 4‑qubit circuit, run through a
    parameterized rotation‑entanglement circuit, and the expectation
    values of Pauli‑Z are extracted as features. These are then
    concatenated and fed into a multi‑task head identical to the
    classical counterpart.
    """

    def __init__(self, patch_size: int = 2, num_patches: int = 14*14,
                 patch_out_dim: int = 4, num_classes: int = 10,
                 n_layers: int = 2, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_out_dim = patch_out_dim
        self.num_patches = num_patches

        # Quantum device with batch support
        self.q_device = qml.device(device_name, wires=4, shots=None)

        # Ansatz parameters: one rotation per qubit per layer
        self.n_layers = n_layers
        self.ansatz_params = nn.Parameter(torch.randn(n_layers, 4))

        # Linear head identical to classical version
        self.classifier = nn.Linear(patch_out_dim * num_patches, num_classes)
        self.reconstructor = nn.Linear(patch_out_dim * num_patches, 28 * 28)

        # Build the qnode
        @qml.qnode(self.q_device, interface="torch")
        def _qnode(patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode data into rotation angles
            for i in range(4):
                qml.RY(patch[i], wires=i)
            # Variational layers
            for l in range(params.shape[0]):
                for i in range(4):
                    qml.RY(params[l, i], wires=i)
                # Entanglement
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
            # Measure expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self._qnode = _qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        logits : torch.Tensor
            Log‑probabilities for the classification task (B, num_classes).
        recon : torch.Tensor
            Reconstructed images (B, 1, 28, 28).
        """
        B = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)
        # Normalize patch values to [0, 1] for stability
        patches = patches / patches.max()

        # Allocate feature tensor
        features = torch.zeros(B, self.num_patches, self.patch_out_dim, device=x.device)

        # Compute quantum features for each sample and patch
        for b in range(B):
            for i in range(self.num_patches):
                patch = patches[b, i, :]
                patch_features = self._qnode(patch, self.ansatz_params)
                features[b, i, :] = patch_features

        flat_features = features.view(B, -1)
        logits = self.classifier(flat_features)
        recon_flat = self.reconstructor(flat_features)
        recon = recon_flat.view(B, 1, 28, 28)

        return F.log_softmax(logits, dim=-1), recon

__all__ = ["QuanvolutionHybrid"]
