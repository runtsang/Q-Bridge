"""Hybrid self‑attention module with classical transformer‑style heads and an optional quantum augmentation.

This module extends the original SelfAttention by providing:
- multi‑head attention with optional weight sharing
- a flag to enable a quantum variational circuit
- a lightweight training loop for the classical branch.

The public API mirrors the original `SelfAttention` but adds the `quantum` and `share_weights` flags.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Dict


class SelfAttentionDual(nn.Module):
    """Self‑attention that can run on classical or quantum back‑ends.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    depth : int, default=4
        Number of attention heads.
    share_weights : bool, default=False
        Whether to share the weight matrices across heads.
    quantum : bool, default=False
        If True, a quantum variational layer is added to each head.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        depth: int = 4,
        share_weights: bool = False,
        quantum: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.share_weights = share_weights
        self.quantum = quantum

        # Classical weight matrices
        if share_weights:
            self.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))
        else:
            self.w_q = nn.Parameter(torch.randn(depth, embed_dim, embed_dim))
            self.w_k = nn.Parameter(torch.randn(depth, embed_dim, embed_dim))
            self.w_v = nn.Parameter(torch.randn(depth, embed_dim, embed_dim))

        # Quantum variational parameters (dummy placeholders for API)
        if quantum:
            self.q_rot = nn.Parameter(torch.randn(depth, embed_dim, 3))
            self.q_ent = nn.Parameter(torch.randn(depth, embed_dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classical multi‑head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.shape
        outputs: List[torch.Tensor] = []

        for h in range(self.depth):
            if self.share_weights:
                wq, wk, wv = self.w_q, self.w_k, self.w_v
            else:
                wq, wk, wv = self.w_q[h], self.w_k[h], self.w_v[h]

            queries = torch.matmul(x, wq)
            keys = torch.matmul(x, wk)
            values = torch.matmul(x, wv)

            scores = torch.softmax(
                torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.embed_dim),
                dim=-1,
            )
            outputs.append(torch.matmul(scores, values))

        out = torch.stack(outputs, dim=1)  # (batch, depth, seq_len, embed_dim)
        out = out.mean(dim=1)  # simple aggregation across heads
        return out

    def run_quantum(
        self,
        backend,
        shots: int = 1024,
    ):
        """Execute a dummy variational circuit that mimics the attention
        pattern.  The circuit is built once per head and sampled
        from the backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Quantum backend to execute the circuit.
        shots : int, default=1024
            Number of shots for the measurement.

        Returns
        -------
        list[dict]
            List of measurement count dictionaries, one per head.
        """
        if not self.quantum:
            raise RuntimeError("Quantum mode not enabled.")

        import qiskit
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        results = []

        for h in range(self.depth):
            qr = QuantumRegister(self.embed_dim, "q")
            cr = ClassicalRegister(self.embed_dim, "c")
            circuit = QuantumCircuit(qr, cr)

            # Simple rotation‑entangle‑rotate structure
            for i in range(self.embed_dim):
                circuit.rx(self.q_rot[h, i, 0].item(), i)
                circuit.ry(self.q_rot[h, i, 1].item(), i)
                circuit.rz(self.q_rot[h, i, 2].item(), i)

            # Entangling gates
            for i in range(self.embed_dim - 1):
                circuit.cx(i, i + 1)

            circuit.measure_all()
            job = qiskit.execute(circuit, backend, shots=shots)
            results.append(job.result().get_counts(circuit))

        return results

    def train_loop(
        self,
        dataloader: Iterable[torch.Tensor],
        loss_fn,
        optimizer,
        epochs: int = 5,
    ) -> None:
        """Very light‑weight training loop that runs the classical branch.

        Parameters
        ----------
        dataloader : iterable
            Iterable yielding batches of input tensors.
        loss_fn : callable
            Loss function that accepts (output, target).
        optimizer : torch.optim.Optimizer
            Optimizer to update the model parameters.
        epochs : int, default=5
            Number of training epochs.
        """
        self.train()
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                out = self.forward(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                optimizer.step()
        self.eval()
