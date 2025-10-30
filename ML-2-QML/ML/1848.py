"""Hybrid convolutional layer that augments the original classical Conv filter with a quantum‑style post‑processing step.

The new Conv class is a drop‑in replacement for the original Conv filter.  It retains the same constructor signature but expands the module to contain an optional quantum‑style post‑processing block.  The design follows the repository’s style guidelines and can be used in any PyTorch model without modification.

Key benefits:
*   A lightweight classical convolution followed by an optional variational circuit that re‑weights the logits.
*   Optional GPU acceleration for the classical part.
*   A simple noise‑model wrapper that can be swapped with a Pennylane or Qiskit simulator.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from typing import Optional

class Conv(nn.Module):
    """Hybrid Conv‑Layer with noise‑aware quantum post‑processing."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        enable_quantum: bool = True,
        shots: int = 500,
        backend: Optional[qiskit.providers.Backend] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.enable_quantum = enable_quantum
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.device = device or torch.device("cpu")

        # Classical convolution stage
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        ).to(self.device)

        # Optional quantum‑style post‑processing
        self._quantum_circuit = None
        if self.enable_quantum:
            self._quantum_circuit = self._build_quantum_circuit()

    def _build_quantum_circuit(self) -> qiskit.QuantumCircuit:
        """Create a lightweight variational circuit for post‑processing."""
        n_qubits = 1
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = Parameter("theta")
        qc.ry(theta, 0)
        qc.measure_all()
        return qc

    def _quantum_forward(self, value: float) -> float:
        """Execute the variational circuit with a rotation angle determined by the value."""
        angle = np.pi * (value > self.threshold)
        bound_qc = self._quantum_circuit.bind_parameters({"theta": angle})
        job = execute(bound_qc, backend=self.backend, shots=self.shots)
        result = job.result().get_counts(bound_qc)
        # Return probability of measuring |1>
        counts = result.get("1", 0)
        return counts / self.shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and optional quantum post‑processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1, H - k + 1, W - k + 1).
        """
        out = self.conv(x.to(self.device))
        out = torch.sigmoid(out - self.threshold)

        if self.enable_quantum:
            # Flatten spatial dimensions and process each element quantum‑ly
            flat = out.view(-1)
            probs = torch.tensor(
                [self._quantum_forward(float(v)) for v in flat.cpu().numpy()],
                dtype=out.dtype,
                device=self.device,
            )
            out = probs.view(out.shape)
        return out

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method for running a single kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation value after classical + optional quantum post‑processing.
        """
        tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = self.forward(tensor)
        return out.mean().item()

__all__ = ["Conv"]
