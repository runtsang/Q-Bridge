"""Quantum Quanvolution: parameter‑shared variational circuit applied to image patches.

The module implements a variational quantum kernel that processes 2×2 image
patches. All patches share the same circuit parameters, enabling efficient
training on larger images. The forward pass returns a concatenated feature
vector of expectation values of Pauli‑Z on each qubit that can be fed into a
classical classifier.

The implementation uses Qiskit’s Aer simulator and the TwoLocal ansatz.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from qiskit import Aer, execute
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import Parameter


class QuanvolutionHybrid(nn.Module):
    """Variational quantum quanvolution filter with shared parameters.

    Parameters
    ----------
    num_qubits : int, default 4
        Number of qubits per patch (2×2 pixels).
    reps : int, default 2
        Number of repetitions of the ansatz layers.
    shots : int, default 1024
        Number of shots for simulation.
    """
    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 2,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        # Parameter‑shared ansatz
        self.circuit = TwoLocal(
            num_qubits,
            ["ry", "rz"],
            "cx",
            reps=reps,
            entanglement="full",
            insert_barriers=False,
        )
        # Parameters for encoding pixel intensities
        self.encoders = [Parameter(f"enc_{i}") for i in range(num_qubits)]
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum kernel to 2×2 patches of the input image.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, H, W) where H and W are divisible by 2.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_patches * num_qubits) containing
            expectation values of Pauli‑Z on each qubit.
        """
        bs, _, h, w = x.shape
        ph, pw = h // 2, w // 2
        num_patches = ph * pw
        # Prepare output tensor
        features = torch.zeros(bs, num_patches * self.num_qubits, device=x.device)
        # Iterate over batch and patches
        patch_idx = 0
        for i in range(ph):
            for j in range(pw):
                # Extract 2×2 patch and flatten
                patch = x[:, 0, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2]
                # Reshape to (batch, 4)
                patch = patch.reshape(bs, -1)
                # For each sample in batch
                for b in range(bs):
                    # Bind parameters to pixel values
                    params = {self.encoders[k]: float(patch[b, k]) for k in range(self.num_qubits)}
                    circ = self.circuit.bind_parameters(params)
                    # Execute
                    job = execute(circ, self.backend, shots=self.shots)
                    result = job.result()
                    counts = result.get_counts(circ)
                    # Compute expectation of Pauli‑Z on each qubit
                    for q in range(self.num_qubits):
                        exp_z = 0.0
                        for bitstring, cnt in counts.items():
                            # Qiskit stores bits in little‑endian order
                            bit = bitstring[self.num_qubits - 1 - q]
                            exp_z += cnt * (1.0 if bit == "0" else -1.0)
                        exp_z /= self.shots
                        features[b, patch_idx * self.num_qubits + q] = exp_z
                patch_idx += 1
        return features


__all__ = ["QuanvolutionHybrid"]
