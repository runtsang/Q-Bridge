"""Quantum counterpart of SelfAttentionHybrid using TorchQuantum."""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class SelfAttentionHybrid(tq.QuantumModule):
    """
    Quantum self‑attention circuit that interleaves:
      * attention rotation & entanglement,
      * kernel encoding (ry gates),
      * classifier ansatz (ry, cz layers).
    Observables are Pauli‑Z on each qubit, yielding classification logits.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2, gamma: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.gamma = gamma

        # Device
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)

        # Parameter vectors
        self.attention_params = tq.ParameterVector("rot", 3 * n_qubits)
        self.entangle_params = tq.ParameterVector("ent", n_qubits - 1)
        self.kernel_params = tq.ParameterVector("kern", n_qubits)
        self.classifier_params = tq.ParameterVector("theta", n_qubits * depth)

    @tq.static_support
    def forward(
        self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Reset device
        q_device.reset_states(x.shape[0])

        # 1. Attention rotation
        for i in range(self.n_qubits):
            q_device.rx(self.attention_params[3 * i], i)
            q_device.ry(self.attention_params[3 * i + 1], i)
            q_device.rz(self.attention_params[3 * i + 2], i)

        # 2. Entanglement
        for i in range(self.n_qubits - 1):
            q_device.crx(self.entangle_params[i], i, i + 1)

        # 3. Kernel encoding (data‑dependent ry)
        for i in range(self.n_qubits):
            q_device.ry(self.kernel_params[i] * x[:, i], i)

        # 4. Classifier ansatz
        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                q_device.ry(self.classifier_params[idx], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                q_device.cz(i, i + 1)

        # 5. Observables (Pauli‑Z on each qubit)
        observables = [
            tq.SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
            for i in range(self.n_qubits)
        ]
        return q_device.expectation(observables)

    def run(
        self,
        backend,
        x: Sequence[float],
        y: Sequence[float],
        shots: int = 1024,
    ):
        """
        Execute the circuit on the specified backend.
        `backend` may be a string name or a TorchQuantum backend instance.
        """
        x_t = torch.as_tensor(x, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        circuit = self.forward(self.q_device, x_t, y_t)
        if isinstance(backend, str):
            backend = tq.backends.get_backend(backend)
        job = backend.execute(circuit, shots=shots)
        return job.result()

__all__ = ["SelfAttentionHybrid"]
