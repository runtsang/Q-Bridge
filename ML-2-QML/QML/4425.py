from __future__ import annotations

import torch
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from typing import Iterable


class SamplerQNN:
    """
    Quantum sampler that returns a probability distribution over 4 outcomes
    using a Qiskit circuit and a TorchQuantum kernel for feature mapping.
    """
    def __init__(self, n_qubits: int = 2) -> None:
        self.n_qubits = n_qubits

        # Build Qiskit circuit
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 4)
        qc = QuantumCircuit(n_qubits)
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        for w, qubit in zip(self.weight_params, range(n_qubits)):
            qc.ry(w, qubit)
        qc.cx(1, 0)
        self.qc = qc

        # Sampler
        self.sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

        # TorchQuantum kernel
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.kernel_ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(4)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities over 4 outcomes.
        x shape (..., 2) – two input angles per sample.
        """
        return self.sampler_qnn(x)

    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Draw samples from the quantum distribution.
        """
        rng = torch.randn(n_samples, 2, device=device)
        probs = self.forward(rng)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a quantum kernel value <x|y> using the ansatz.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        for info in self.kernel_ansatz:
            params = x[:, info["input_idx"]]
            tqf.ry(self.q_device, params, wires=info["wires"])
        for info in reversed(self.kernel_ansatz):
            params = -y[:, info["input_idx"]]
            tqf.ry(self.q_device, params, wires=info["wires"])
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute Gram matrix between two sets of feature vectors.
        """
        return torch.stack([torch.stack([self.kernel(x, y) for y in b]) for x in a])


__all__ = ["SamplerQNN"]
