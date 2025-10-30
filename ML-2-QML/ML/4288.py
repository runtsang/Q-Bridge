import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class QuantumKernel:
    """
    Compute a quantum kernel via a parameterised Ry‑circuit on a statevector
    simulator.  Parameters are optionally clipped to a user‑supplied bound.
    """
    def __init__(self, clip: bool = True, bound: float = 5.0):
        self.backend = Aer.get_backend("statevector_simulator")
        self.clip = clip
        self.bound = bound

    def _statevector(self, params: torch.Tensor) -> np.ndarray:
        n = params.shape[0]
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        for i, theta in enumerate(params):
            circuit.ry(float(theta), i)
        job = execute(circuit, self.backend)
        return np.array(job.result().get_statevector(circuit))

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.clip:
            x = torch.clamp(x, -self.bound, self.bound)
            y = torch.clamp(y, -self.bound, self.bound)
        sv_x = self._statevector(x)
        sv_y = self._statevector(y)
        overlap = np.vdot(sv_x, sv_y)
        return torch.tensor(abs(overlap)**2, dtype=torch.float32)

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer that uses the QuantumKernel to compute
    a similarity feature map between the input batch and a set of support
    vectors.  The output is an affine transformation of the kernel
    matrix, mirroring the scaling/shift logic from the fraud‑detection
    reference.
    """
    def __init__(self, support_vectors: torch.Tensor, clip: bool = True):
        super().__init__()
        self.support_vectors = support_vectors
        self.n_support = support_vectors.shape[0]
        self.weight = nn.Parameter(torch.randn(self.n_support))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))
        self.kernel_obj = QuantumKernel(clip=clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        k = torch.empty(batch, self.n_support, device=x.device)
        for i in range(batch):
            for j in range(self.n_support):
                k[i, j] = self.kernel_obj.kernel(x[i], self.support_vectors[j])
        out = k @ self.weight
        out = out * self.scale + self.shift
        return out

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Return the Gram matrix between two sets of feature vectors.
        """
        n_a, n_b = a.shape[0], b.shape[0]
        out = torch.empty(n_a, n_b, device=a.device)
        for i in range(n_a):
            for j in range(n_b):
                out[i, j] = self.kernel_obj.kernel(a[i], b[j])
        return out

__all__ = ["HybridFullyConnectedLayer", "QuantumKernel"]
