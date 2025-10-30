import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Iterable, List

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
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
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit."""
    def __init__(self, circuit: tq.QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = tq.StateVector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridEstimator(QuantumEstimator):
    """Hybrid quantum‑classical estimator: a quantum kernel feeding a classical linear head."""
    def __init__(self, circuit: tq.QuantumCircuit, num_classes: int = 2):
        super().__init__(circuit)
        self.kernel = Kernel()
        self.linear = torch.nn.Linear(1, num_classes)

    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        raw = super().evaluate(observables, parameter_sets)
        flat = [item[0] if isinstance(item, list) else item for item in raw]
        tensor = torch.tensor(flat).unsqueeze(1)
        logits = self.linear(tensor)
        return logits

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumEstimator",
    "HybridEstimator",
]
