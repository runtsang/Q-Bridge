import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, execute, Aer

class HybridSelfAttention(nn.Module):
    """
    Hybrid classical–quantum self‑attention module.

    Parameters
    ----------
    embed_dim : int, default=4
        Dimensionality of the input embeddings.
    n_qubits : int, default=4
        Number of qubits used in the quantum circuit.
    kernel_size : int, default=2
        Size of the convolution kernel applied to the input.
    """
    def __init__(self, embed_dim: int = 4, n_qubits: int = 4, kernel_size: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size

        # Classical attention projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Convolutional pre‑processor
        self.conv = nn.Conv2d(1, 1, kernel_size, bias=True)

        # Quantum circuit template
        self._base_qc = self._build_qc()
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_qc(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(0.0, i)
            qc.ry(0.0, i)
            qc.rz(0.0, i)
        for i in range(self.n_qubits - 1):
            qc.crx(0.0, i, i + 1)
        qc.measure_all()
        return qc

    def _run_quantum(self, params: np.ndarray, shots: int = 512) -> float:
        qc = self._base_qc.copy()
        for i in range(self.n_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        job = execute(qc, self.backend, shots=shots)
        result = job.result().get_counts(qc)
        # Average number of |1> outcomes per qubit
        total_ones = sum(int(bit) for key in result for bit in key) * result[key]
        return total_ones / (shots * self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid attention output.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq, _ = x.shape

        # Convolutional pre‑processor
        x_conv = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, embed_dim, seq)
        conv_out = torch.sigmoid(
            self.conv(x_conv[:, :, :self.kernel_size, :self.kernel_size]).mean()
        )

        # Classical self‑attention
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        classical_out = scores @ V

        # Quantum attention
        params = np.random.rand(self.n_qubits * 3)
        qprob = self._run_quantum(params)

        # Combine all signals
        out = classical_out + qprob * conv_out
        return out
