"""Combined quantum kernel and estimator module."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Iterable, Callable, List

# --------------------------------------------------------------------------- #
# Quantum kernel utilities
# --------------------------------------------------------------------------- #
class QuantumKernalAnsatz(tq.QuantumModule):
    """Programmable ansatz that encodes two input vectors with Ry rotations."""
    def __init__(self, gate_list):
        super().__init__()
        self.gate_list = gate_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.gate_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.gate_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernalAnsatz(
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between two lists of vectors."""
    kernel = QuantumKernel()
    return kernel(torch.stack(a), torch.stack(b)).cpu().numpy()

# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Simple parametric circuit that returns the expectation of Z on the first qubit."""
    def __init__(self, n_wires: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        self.q_device.reset_states(1)
        tq.ry(thetas_tensor, wires=[0], q_device=self.q_device)
        # Measure Z expectation
        state = self.q_device.states.view(-1)
        probs = torch.abs(state) ** 2
        expectation = probs[0] - probs[1]
        return expectation.detach().numpy()

# --------------------------------------------------------------------------- #
# Quantum estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimatorQuantum:
    """Evaluate expectation values of Pauli observables for a parametric circuit."""
    def __init__(self, circuit: QuantumFullyConnectedLayer) -> None:
        self.circuit = circuit

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            exp_values = self.circuit.run(params)
            row: List[float] = []
            for obs in observables:
                val = obs(exp_values)
                row.append(float(val))
            results.append(row)
        return results

class FastEstimatorQuantum(FastBaseEstimatorQuantum):
    """Adds Gaussian shot noise to quantum expectation values."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Combined shared model (quantum)
# --------------------------------------------------------------------------- #
class SharedKernelModel:
    """Unified interface that supports quantum kernels, a fully‑connected quantum layer,
    and a noisy estimator."""
    def __init__(
        self,
        n_wires: int = 4,
        noise_shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.kernel = QuantumKernel(n_wires)
        self.fcl = QuantumFullyConnectedLayer()
        self.estimator = FastEstimatorQuantum(self.fcl)
        self.noise_shots = noise_shots
        self.seed = seed

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return self.estimator.evaluate(
            observables, parameter_sets, shots=self.noise_shots, seed=self.seed
        )

__all__ = ["SharedKernelModel"]
