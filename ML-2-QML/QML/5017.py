"""Hybrid estimator integrating quantum circuits, kernels, and quanvolution filters."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Tuple

from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.random import random_circuit

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ------------------------------------------------------------------
# Quantum classifier circuit factory
# ------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ------------------------------------------------------------------
# Quantum kernel using TorchQuantum
# ------------------------------------------------------------------
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

    def __init__(self):
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
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Quanvolution filter
# ------------------------------------------------------------------
def Conv():
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""

        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = QuantumCircuit(self.n_qubits)
            self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the quantum circuit on classical data.

            Args:
                data: 2D array with shape (kernel_size, kernel_size).

            Returns:
                float: average probability of measuring |1> across qubits.
            """
            data = np.reshape(data, (1, self.n_qubits))

            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = execute(self._circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    backend = Aer.get_backend("qasm_simulator")
    filter_size = 2
    circuit = QuanvCircuit(filter_size, backend, shots=100, threshold=127)
    return circuit

# ------------------------------------------------------------------
# Hybrid quantum estimator
# ------------------------------------------------------------------
class HybridEstimator:
    """Evaluate a quantum circuit, optionally with Gaussian shot noise, and expose
    kernel and quanvolution utilities."""

    def __init__(self, circuit: QuantumCircuit | None = None, backend: object | None = None, shots: int | None = None) -> None:
        self.circuit = circuit
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        if self.circuit is None:
            raise ValueError("HybridEstimator requires a circuit.")
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed=42)
        for values in parameter_sets:
            bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            if self.shots is not None:
                # Inject Gaussian noise to mimic shot noise
                noisy_row = [float(rng.normal(val.real, max(1e-6, 1 / self.shots))) for val in row]
                results.append(noisy_row)
            else:
                results.append(row)
        return results

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Wrap the quantum kernel defined above."""
        return kernel_matrix(a, b)

    @staticmethod
    def conv_filter(kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        """Return a quanvolution filter instance."""
        return Conv()


__all__ = ["HybridEstimator"]
