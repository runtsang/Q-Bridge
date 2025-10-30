import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Iterable, List, Union
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --- Qiskit fast estimator ----------------------------------------------------
class FastBaseEstimatorQiskit:
    """
    Evaluate expectation values of observables for a parametrized circuit.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimatorQiskit(FastBaseEstimatorQiskit):
    """
    Adds Gaussian shot noise to the deterministic estimator.
    """
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(v.real, max(1e-6, 1 / shots)),
                    rng.normal(v.imag, max(1e-6, 1 / shots)),
                )
                for v in row
            ]
            noisy.append(noisy_row)
        return noisy

# --- Quantum kernel with TorchQuantum -----------------------------------------
class KernalAnsatz(tq.QuantumModule):
    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    def __init__(self, n_wires: int = 4, ansatz_params: List[dict] | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            ansatz_params or [
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

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4) -> np.ndarray:
    kernel = Kernel(n_wires=n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --- Combined QML class --------------------------------------------------------
class QuantumKernelMethod:
    """
    Hybrid quantum kernel interface that exposes a TorchQuantum kernel and a Qiskit
    estimator for expectation values of parametrized circuits.
    """
    def __init__(
        self,
        n_wires: int = 4,
        ansatz_params: List[dict] | None = None,
        circuit: QuantumCircuit | None = None,
    ) -> None:
        self.n_wires = n_wires
        self.kernel = Kernel(n_wires=n_wires, ansatz_params=ansatz_params)
        self.circuit = circuit
        if circuit is not None:
            self.estimator = FastEstimatorQiskit(circuit)
        else:
            self.estimator = None

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return quantum_kernel_matrix(a, b, self.n_wires)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if self.estimator is None:
            raise ValueError("No circuit provided for evaluation.")
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "QuantumKernelMethod",
    "KernalAnsatz",
    "Kernel",
    "quantum_kernel_matrix",
    "FastBaseEstimatorQiskit",
    "FastEstimatorQiskit",
]
