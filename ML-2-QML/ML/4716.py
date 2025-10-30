import torch
import torch.nn as nn
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import Aer, QuantumCircuit

class UnifiedQuantumHybridLayer(nn.Module):
    """
    Hybrid layer combining a classical dense block, a classical convolutional filter,
    and a quantum variational block.  The module exposes a ``run`` method that
    mimics the API of the seed modules and returns a NumPy array of the
    concatenated outputs.
    """
    def __init__(self,
                 n_features: int,
                 conv_kernel: int = 2,
                 depth: int = 2,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # Classical dense block
        self.dense = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh()
        )

        # Classical convolutional filter
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=conv_kernel,
            bias=True
        )

        # Quantum variational parameters
        self.theta = nn.Parameter(torch.randn(depth, dtype=torch.float32, device=self.device))
        self.q_params = ParameterVector("theta", depth)

        # Build a reusable template circuit
        self.circuit_template = self._build_qcircuit()
        self.backend = Aer.get_backend("statevector_simulator")
        self.depth = depth

    def _build_qcircuit(self) -> QuantumCircuit:
        """Create a 1‑qubit circuit with Ry layers parameterized by ``self.q_params``."""
        qc = QuantumCircuit(1)
        qc.h(0)
        for p in self.q_params:
            qc.ry(p, 0)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that concatenates the outputs of the three blocks.
        * x: Tensor of shape (batch, n_features)
        * returns: Tensor of shape (batch, 3)
        """
        # Dense block
        d = self.dense(x).mean(dim=1, keepdim=True)  # (batch,1)

        # Convolutional block
        img = x[:, :self.conv.kernel_size**2].view(-1, 1, self.conv.kernel_size, self.conv.kernel_size)
        conv_out = self.conv(img).view(-1, 1)  # (batch,1)
        conv_out = torch.sigmoid(conv_out - self.conv.bias)  # thresholding

        # Quantum block
        bind_dict = {self.q_params[i]: self.theta[i].item() for i in range(len(self.theta))}
        bound = self.circuit_template.bind_parameters(bind_dict)
        sv = self.backend.run(bound).result().get_statevector()
        # Expectation of Z: |0>^2 - |1>^2
        exp_z = sv[0].real - sv[1].real
        q_out = torch.full((x.size(0), 1), exp_z, device=self.device, dtype=torch.float32)

        # Concatenate
        out = torch.cat([d, conv_out, q_out], dim=1)
        return out

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compatibility method that accepts a NumPy array of variational parameters
        and returns the concatenated output as a 1‑D NumPy array.
        """
        with torch.no_grad():
            theta_t = torch.tensor(thetas, dtype=torch.float32, device=self.device)
            self.theta.data = theta_t
            dummy = torch.zeros(1, self.dense[0].in_features, device=self.device)
            out = self.forward(dummy)
        return out.squeeze().cpu().numpy()
