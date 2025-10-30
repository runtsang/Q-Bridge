"""Hybrid quantum kernel mixing a variational kernel and a sampler similarity."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler

class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz encoding classical data via Ry rotations."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class SamplerQNN:
    """Quantum sampler circuit producing a probability distribution."""
    def __init__(self, n_wires: int = 2):
        self.n_wires = n_wires
        self.inputs = ParameterVector("input", n_wires)
        self.weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(n_wires)
        for i in range(n_wires):
            qc.ry(self.inputs[i], i)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        self.circuit = qc
        self.sampler = StatevectorSampler()
        self.compiled_sampler = self.sampler.compile(self.circuit)

    def __call__(self, input_params: np.ndarray) -> np.ndarray:
        """Return probability vector for given input parameters."""
        weights = np.zeros(4)
        param_dict = {"input": input_params, "weight": weights}
        result = self.compiled_sampler(param_dict)
        # Extract probabilities from the statevector
        statevector = result[0] if isinstance(result, (list, tuple)) else result.statevector
        probs = np.abs(statevector)**2
        return probs

class HybridKernelMethod:
    """Quantum hybrid kernel mixing a variational kernel and a sampler similarity."""
    def __init__(self, n_wires: int = 4, weight_rbf: float = 0.5, weight_sampler: float = 0.5):
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.sampler = SamplerQNN(n_wires=2)
        self.weight_rbf = weight_rbf
        self.weight_sampler = weight_sampler

    def _sampler_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dot product of sampler output probability vectors."""
        sx = torch.tensor(self.sampler(x.numpy()), dtype=torch.float32)
        sy = torch.tensor(self.sampler(y.numpy()), dtype=torch.float32)
        return torch.dot(sx, sy) / (torch.norm(sx) * torch.norm(sy) + 1e-12)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return weighted mixture of variational kernel and sampler similarity."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        rbf_val = torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)
        samp_val = self._sampler_similarity(x.squeeze(0), y.squeeze(0))
        return self.weight_rbf * rbf_val + self.weight_sampler * samp_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences."""
        mat = torch.stack([torch.stack([self.forward(x, y).squeeze() for y in b]) for x in a])
        return mat.detach().cpu().numpy()

__all__ = ["HybridKernelMethod"]
