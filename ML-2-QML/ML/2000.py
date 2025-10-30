import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


class ConvEnhanced(nn.Module):
    """
    Hybrid convolution module with optional quantum back‑end.
    """
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "same",
        groups: int = 1,
        use_quantum: bool = False,
        threshold: float = 0.0,
        vqc_depth: int = 2,
        vqc_layers: int = 2,
        shots: int = 512,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # Classical depth‑wise separable conv
        self.classical = nn.Conv2d(
            1,
            1,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding == "same" else 0,
            groups=groups,
            bias=True,
        )

        # Learnable threshold for binarisation
        self.threshold = nn.Parameter(torch.tensor(float(threshold), dtype=torch.float32))

        self.use_quantum = use_quantum
        self.shots = shots

        if use_quantum:
            self.n_qubits = kernel_size ** 2
            self.backend = Aer.get_backend("qasm_simulator")
            self.circuit = self._build_vqc(self.n_qubits, vqc_depth, vqc_layers)

    def _build_vqc(self, n_qubits: int, depth: int, layers: int):
        """
        Build a simple variational quantum circuit.
        The circuit contains alternating RX and RZ on each qubit,
        followed by a 2‑qubit entangling pattern.
        """
        qc = QuantumCircuit(n_qubits)
        self.theta_rx = [Parameter(f"theta_rx_{i}") for i in range(n_qubits)]
        self.theta_rz = [Parameter(f"theta_rz_{i}") for i in range(n_qubits)]

        for _ in range(depth):
            for i in range(n_qubits):
                qc.rx(self.theta_rx[i], i)
                qc.rz(self.theta_rz[i], i)
            # Entangling pattern: CX between adjacent qubits
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                qc.cx(i, i + 1)

        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean filter response.
        """
        if not self.use_quantum:
            out = self.classical(x)
            return out.mean()

        # Quantum branch
        # Extract patches
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )  # shape (B, kernel_size*kernel_size, L)

        B, _, L = patches.shape
        probs = []

        for b in range(B):
            for l in range(L):
                patch = patches[b, :, l].detach().cpu().numpy()
                # Bind parameters
                param_bind = {}
                for i, val in enumerate(patch):
                    param_bind[self.theta_rx[i]] = np.pi if val > self.threshold.item() else 0.0
                    param_bind[self.theta_rz[i]] = 0.0  # fixed for simplicity
                job = execute(
                    self.circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[param_bind],
                )
                result = job.result()
                counts = result.get_counts(self.circuit)
                # Compute probability of measuring |1> across all qubits
                ones = 0
                for bitstring, count in counts.items():
                    ones += bitstring.count("1") * count
                prob = ones / (self.shots * self.n_qubits)
                probs.append(prob)

        mean_prob = np.mean(probs)
        return torch.tensor(mean_prob, dtype=torch.float32)
