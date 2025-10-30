import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQuantumFullyConnected(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in classical, quantum,
    or hybrid mode.  The classical part is a dense projection.  The
    quantum part is a parameter‑shared variational circuit approximated
    by a differentiable analytic function.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int = 4,
                 n_qubits: int = 4,
                 use_quantum: bool = False,
                 device: str = "cpu",
                 shots: int = 1024):
        super().__init__()
        self.use_quantum = use_quantum
        self.device = device
        self.shots = shots
        self.n_qubits = n_qubits

        # Classical linear mapping
        self.classical = nn.Linear(in_features, out_features)

        # Encoder that maps input features to quantum parameters
        self.encoder = nn.Linear(in_features, n_qubits * 3)

        # Parameters for the quantum circuit (one RX, RY, RZ per qubit)
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(3, device=self.device)) for _ in range(n_qubits)]
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass. If `use_quantum` is False, returns the classical
        output.  Otherwise, returns the sum of the classical output and
        a quantum‑inspired expectation value computed analytically.
        """
        out = self.classical(x)

        if not self.use_quantum:
            return out

        # Encode input into parameters for the quantum circuit
        encoded = self.encoder(x)  # shape (batch, n_qubits*3)
        encoded = encoded.view(x.shape[0], self.n_qubits, 3)  # (batch, qubits, 3)

        # Compute quantum expectation analytically
        expectation = self._analytic_expectation(encoded)

        # Combine classical and quantum outputs
        return out + expectation

    def _analytic_expectation(self, encoded: torch.Tensor):
        """
        Approximate expectation value of a variational circuit with
        RX, RY, RZ gates followed by a CNOT chain.  For each sample
        we compute a simple differentiable function that captures
        the non‑trivial dependence on the parameters:
            E = mean( cos(rx) * sin(ry) * cos(rz) )
        """
        rx = encoded[:, :, 0]
        ry = encoded[:, :, 1]
        rz = encoded[:, :, 2]
        e = torch.cos(rx) * torch.sin(ry) * torch.cos(rz)
        e = torch.mean(e, dim=1, keepdim=True)
        return e
