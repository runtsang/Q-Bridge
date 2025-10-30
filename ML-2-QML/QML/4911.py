import torch
import torchquantum as tq
from.QuantumKernelMethod import Kernel as QuantumKernel

class ConvHybrid(tq.QuantumModule):
    """
    Quantum‑augmented convolutional module that mirrors the classical
    ConvHybrid API.  Each 2×2 patch is encoded into 4 qubits, processed
    by a small variational circuit, and measured to obtain a 4‑dimensional
    feature vector.  The module optionally evaluates a quantum kernel
    against a set of reference feature vectors.
    """
    def __init__(
        self,
        patch_size: int = 2,
        n_wires: int = 4,
        use_kernel: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(
            n_ops=20, wires=list(range(n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.use_kernel = use_kernel
        self.kernel = QuantumKernel() if use_kernel else None

    def _patches(self, img: torch.Tensor) -> torch.Tensor:
        """
        Split image into non‑overlapping patches of size (patch_size, patch_size).
        Returns tensor of shape (B, C, N, patch_size, patch_size).
        """
        B, C, H, W = img.shape
        ph, pw = self.patch_size, self.patch_size
        patches = img.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(B, C, -1, ph, pw)
        return patches

    def forward(self, x: torch.Tensor, refs: torch.Tensor | None = None):
        """
        Args:
            x: Tensor (B, C, H, W)
            refs: Optional tensor of reference feature vectors for kernel evaluation.
        Returns:
            feat: Tensor (B, n_wires)
            kernel_mat: Optional kernel matrix (B, N) if refs provided.
        """
        patches = self._patches(x)
        B, C, N, ph, pw = patches.shape
        device = x.device
        flat = patches.view(B * C * N, ph * pw)

        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=B * C * N,
            device=device,
            record_op=False,
        )

        # Encode each pixel into an RY rotation on its dedicated qubit.
        for i in range(self.n_wires):
            gate = tq.RY(has_params=False)
            gate(qdev, wires=i, params=flat[:, i])

        # Apply a light random layer to entangle the qubits.
        self.random_layer(qdev)

        # Measure all qubits to obtain expectation values.
        probs = self.measure(qdev).reshape(B, C, N, self.n_wires)
        # Average over channels and spatial locations to produce a feature vector.
        feat = probs.mean(dim=[1, 2])  # (B, n_wires)

        if self.use_kernel and refs is not None:
            km = torch.stack([self.kernel(feat[i], refs) for i in range(feat.size(0))])
            return feat, km
        return feat

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel Gram matrix between two batches of feature vectors.
        """
        return torch.stack([self.kernel(a[i], b) for i in range(a.size(0))])

__all__ = ["ConvHybrid"]
