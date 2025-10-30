"""Quantum kernel that emulates transformer‑style encoding and returns state overlap."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumTransformerAnsatz(tq.QuantumModule):
    """
    A simple quantum circuit that mimics a transformer embedding.
    Each token is encoded on a contiguous block of qubits; a CNOT ladder
    entangles neighbouring tokens, providing a rudimentary attention‑like
    interaction.
    """
    def __init__(self,
                 n_wires_per_token: int,
                 n_tokens: int) -> None:
        super().__init__()
        self.n_wires_per_token = n_wires_per_token
        self.n_tokens = n_tokens
        self.total_wires = n_wires_per_token * n_tokens
        # Encoder that applies RX to each qubit with the corresponding input value
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(self.total_wires)
            ]
        )
        # Entanglement ladder across all wires
        self.entangle = [
            {"func": "cnot", "wires": [i, i + 1]}
            for i in range(self.total_wires - 1)
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_tokens, n_wires_per_token).
        q_device : tq.QuantumDevice
            Quantum device that will be reused for all samples.
        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape (batch, total_wires).
        """
        # Flatten tokens into a single batch dimension
        batch = x.shape[0] * x.shape[1]
        flat = x.reshape(batch, self.total_wires)
        self.encoder(q_device, flat)
        for step in self.entangle:
            func_name_dict[step["func"]](q_device, wires=step["wires"])
        return self.measure(q_device).reshape(x.shape[0], -1)


class TransformerQuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that encodes two sequences with `QuantumTransformerAnsatz`
    and returns the magnitude of their state overlap.
    """
    def __init__(self,
                 n_wires_per_token: int = 4,
                 n_tokens: int = 8,
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_wires_per_token = n_wires_per_token
        self.n_tokens = n_tokens
        self.q_device = tq.QuantumDevice(n_wires=n_wires_per_token * n_tokens)
        self.ansatz = QuantumTransformerAnsatz(n_wires_per_token, n_tokens)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of sequences.
        Each sequence is expected to have shape (batch, n_tokens, n_wires_per_token).
        """
        # Encode x
        self.ansatz(self.q_device, x)
        ket_x = self.q_device.states.clone()
        # Reset device before encoding y
        self.q_device.reset_states(x.shape[0])
        # Encode y with negated angles to compute overlap
        self.ansatz(self.q_device, -y)
        ket_y = self.q_device.states.clone()
        # Overlap magnitude
        overlap = torch.abs((ket_x.conj() * ket_y).sum(-1))
        return self.gamma * overlap

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  kernel: TransformerQuantumKernel) -> torch.Tensor:
    """
    Compute the Gram matrix for two collections of sequences using the quantum kernel.
    Each evaluation runs on the shared quantum device; the function is
    vectorised over the outer loop for convenience.
    """
    rows = []
    for x in a:
        row = []
        for y in b:
            row.append(kernel(x.unsqueeze(0), y.unsqueeze(0)))
        rows.append(torch.stack(row))
    return torch.stack(rows)

__all__ = ["QuantumTransformerAnsatz", "TransformerQuantumKernel", "kernel_matrix"]
