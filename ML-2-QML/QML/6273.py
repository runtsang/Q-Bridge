"""Hybrid quantum kernel with optional Qiskit SamplerQNN re‑weighting.

The class `QuantumKernelMethod` evaluates a fixed TorchQuantum ansatz
to compute a quantum kernel and can optionally re‑weight the result
using a Qiskit SamplerQNN.  The implementation preserves the
original API while adding data‑dependent weighting, enabling richer
similarity measures."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Programmable list of gates that encodes classical data."""
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


class Kernel(tq.QuantumModule):
    """Fixed four‑qubit ansatz used to compute the quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class SamplerQNNQiskit:
    """Wrapper around Qiskit’s SamplerQNN that returns probabilities."""
    def __init__(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler

        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
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
        self.sampler_qnn = SamplerQNN(circuit=qc, input_params=inputs,
                                     weight_params=weights, sampler=sampler)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Return probability vector for each input sample."""
        return self.sampler_qnn.sample(inputs)


class QuantumKernelMethod:
    """Hybrid quantum kernel that can be re‑weighted by a Qiskit SamplerQNN."""
    def __init__(self, use_sampler: bool = False):
        self.kernel = Kernel()
        self.use_sampler = use_sampler
        self.sampler = SamplerQNNQiskit() if use_sampler else None

    def _sample_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampler is None:
            return torch.ones(1, device=x.device)
        probs = self.sampler(x.numpy())
        return torch.tensor(probs[0][0], device=x.device).unsqueeze(-1)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.kernel(x, y)
        weight = self._sample_weight(x)
        return k * weight

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.sampler is None:
            raise RuntimeError("Sampler not enabled for this instance.")
        return torch.tensor(self.sampler(inputs.numpy()), device=inputs.device)


__all__ = ["KernalAnsatz", "Kernel", "SamplerQNNQiskit", "QuantumKernelMethod"]
