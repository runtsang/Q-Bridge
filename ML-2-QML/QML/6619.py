"""Hybrid kernel‑classifier combining a TorchQuantum kernel and a Qiskit variational circuit."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumRBFKernel(tq.QuantumModule):
    """Quantum kernel that encodes data via Ry rotations."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = self._create_ansatz()

    def _create_ansatz(self) -> tq.QuantumModule:
        """Build a simple list‑based ansatz for encoding."""
        return tq.QuantumModule(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the overlap as a scalar."""
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridKernelClassifier:
    """Hybrid kernel‑classifier with a quantum kernel and a Qiskit variational circuit."""

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        self.kernel = QuantumRBFKernel(n_wires)
        self.circuit, self.encoding, self.weights, self.observables = self._build_classifier_circuit(
            n_wires, depth
        )

    def _build_classifier_circuit(
        self, num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Create a layered ansatz with explicit encoding and variational parameters."""
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

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, [encoding], [weights], observables

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the quantum kernel."""
        return np.array([[self.kernel.compute(x, y).item() for y in b] for x in a])

    def evaluate_circuit(self, inputs: torch.Tensor) -> np.ndarray:
        """Run the variational circuit on a batch of inputs."""
        # Simple simulation using Qiskit Aer (placeholder)
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        job = simulator.run(self.circuit.bind_parameters({self.circuit.parameters[i]: inputs[i] for i in range(len(inputs))}))
        result = job.result()
        return result.get_counts(self.circuit)

__all__ = ["HybridKernelClassifier"]
