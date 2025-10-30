"""Hybrid quantum kernel method with optional Qiskit sampler.

The implementation uses TorchQuantum for the kernel evaluation and
provides a thin wrapper around Qiskit’s SamplerQNN for data
generation.  The class is compatible with the legacy interface
while exposing a richer, variational‑circuit based kernel.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler


class QuantumKernAlAnsatz(tq.QuantumModule):
    """Parameterised ansatz that encodes two inputs via Ry rotations."""
    def __init__(self, gate_defs: list[dict]) -> None:
        super().__init__()
        self.gate_defs = gate_defs

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # encode first vector
        for g in self.gate_defs:
            params = x[:, g["input_idx"]]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)
        # un‑encode second vector with negative parameters
        for g in reversed(self.gate_defs):
            params = -y[:, g["input_idx"]]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)


class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel with optional Qiskit sampler."""
    def __init__(self,
                 n_wires: int = 4,
                 use_sampler: bool = False) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # a minimal, yet flexible, Ry‑only ansatz
        self.ansatz = QuantumKernAlAnsatz([
            {"input_idx": 0, "func": "ry", "wires": 0},
            {"input_idx": 1, "func": "ry", "wires": 1},
            {"input_idx": 2, "func": "ry", "wires": 2},
            {"input_idx": 3, "func": "ry", "wires": 3},
        ])
        self.use_sampler = use_sampler
        if use_sampler:
            # build a Qiskit SamplerQNN circuit
            inputs = ParameterVector("x", 2)
            weights = ParameterVector("w", 4)
            qc = QuantumCircuit(2)
            qc.ry(inputs[0], 0)
            qc.ry(inputs[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[0], 0)
            qc.ry(weights[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[2], 0)
            qc.ry(weights[3], 1)
            sampler = StatevectorSampler()
            self.sampler_qnn = QiskitSamplerQNN(
                circuit=qc,
                input_params=inputs,
                weight_params=weights,
                sampler=sampler,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix via the quantum kernel."""
        mat = torch.stack([self.forward(a[i].unsqueeze(0), b[j].unsqueeze(0))
                           for i in range(a.shape[0]) for j in range(b.shape[0])])
        return mat.reshape(a.shape[0], b.shape[0]).cpu().numpy()

    def sample(self, inputs: torch.Tensor) -> np.ndarray:
        """Delegate to the Qiskit sampler if enabled."""
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled in this instance.")
        # convert torch tensor to numpy array for Qiskit
        data = inputs.detach().cpu().numpy()
        return self.sampler_qnn.predict(data)


__all__ = ["HybridKernelMethod", "QuantumKernAlAnsatz"]
