"""Hybrid quantum kernel module with quantum backends.

The module defines :class:`HybridQuantumKernel` which exposes a unified interface
for computing quantum kernels, evaluating fast estimators, sampling from
parameterized quantum networks, and running a fully‑connected quantum‑style layer.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Callable

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ---------- Quantum kernel ----------
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

class QuantumKernel(tq.QuantumModule):
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
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------- Fast estimator ----------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

# ---------- Sampler network ----------
def SamplerQNN() -> QuantumCircuit:
    """A simple example of a parameterized quantum circuit for a SamplerQNN."""
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)

    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    return qc2

# ---------- Fully‑connected layer ----------
def FCL() -> QuantumCircuit:
    """A simple example of a parameterized quantum circuit for a fully connected layer."""
    class QuantumCircuitWrapper:
        def __init__(self, n_qubits, backend, shots):
            self._circuit = QuantumCircuit(n_qubits)
            self.theta = ParameterVector("theta", 1)
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta[0], range(n_qubits))
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta[0]: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    simulator = qiskit.Aer.get_backend("qasm_simulator")
    circuit = QuantumCircuitWrapper(1, simulator, 100)
    return circuit

# ---------- Hybrid kernel class ----------
class HybridQuantumKernel:
    """
    Unified interface for quantum kernel operations.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits for the quantum kernel.
    use_sampler : bool, optional
        Whether to expose a sampler network.
    use_fcl : bool, optional
        Whether to expose a fully‑connected layer.
    """
    def __init__(self, n_wires: int = 4, use_sampler: bool = False, use_fcl: bool = False) -> None:
        self.n_wires = n_wires
        self.quantum_kernel = QuantumKernel()
        self.use_sampler = use_sampler
        self.use_fcl = use_fcl
        if use_sampler:
            self.sampler = SamplerQNN()
        if use_fcl:
            self.fcl = FCL()

    # Quantum kernel
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.quantum_kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    # Fast estimator wrapper
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.sampler if self.use_sampler else self.fcl)
        return estimator.evaluate(observables, parameter_sets)

    # Sampler
    def sample(self, parameter_values: Sequence[float]) -> np.ndarray:
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled for this kernel.")
        return self.sampler.run(parameter_values)

    # Fully‑connected layer
    def run_fcl(self, thetas: Sequence[float]) -> np.ndarray:
        if not self.use_fcl:
            raise RuntimeError("Fully‑connected layer not enabled.")
        return self.fcl.run(thetas)

__all__ = [
    "HybridQuantumKernel",
    "KernalAnsatz",
    "QuantumKernel",
    "kernel_matrix",
    "FastBaseEstimator",
    "SamplerQNN",
    "FCL",
]
