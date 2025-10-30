"""Combined quantum estimator with circuit, filter and kernel support."""

from __future__ import annotations

import numpy as np
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Quantum filter (quanvolution)
# ----------------------------------------------------------------------
class QuanvCircuit:
    """Parametrised circuit that acts as a quantum filter."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the filter on classical 2‑D data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# ----------------------------------------------------------------------
# Quantum kernel ansatz
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two inputs into a quantum device and returns an overlap."""
    def __init__(self, func_list: list[dict]) -> None:
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
    """Quantum kernel that computes the absolute overlap of two encoded states."""
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
    """Compute Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Combined quantum estimator
# ----------------------------------------------------------------------
ScalarObservable = Callable[[Union[Statevector, np.ndarray, None]], torch.Tensor | float | complex]

class FastBaseEstimatorGen152:
    """Unified quantum estimator that can evaluate a circuit, a filter or a quantum kernel.
    Supports optional shot‑noise on expectation values."""
    def __init__(self, circuit: QuantumCircuit | Kernel | None = None) -> None:
        self.circuit = circuit
        self.kernel = circuit if isinstance(circuit, Kernel) else None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if self.circuit is None or not isinstance(self.circuit, QuantumCircuit):
            raise ValueError("No quantum circuit to bind.")
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | Callable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            if isinstance(self.circuit, QuantumCircuit):
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) if isinstance(obs, BaseOperator) else obs(state) for obs in observables]
            elif isinstance(self.circuit, Kernel):
                # Kernel evaluation: compute kernel between the vector and itself
                row = [self.circuit(torch.tensor(values, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)).item()]
            elif hasattr(self.circuit, "run"):
                # Quantum filter
                row = [self.circuit.run(np.asarray(values))]
            else:
                raise TypeError("Unsupported circuit type.")
            results.append(row)

        # Apply shot noise if requested
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(rng.normal(mean.real, max(1e-6, 1 / shots))) + 1j * rng.normal(mean.imag, max(1e-6, 1 / shots))
                if isinstance(mean, complex) else rng.normal(mean, max(1e-6, 1 / shots))
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuanvCircuit", "KernalAnsatz", "Kernel", "kernel_matrix", "FastBaseEstimatorGen152"]
