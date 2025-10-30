"""UnifiedHybridLayer - Quantum implementation.

This module provides a quantum version of the hybrid layer.  It supports:
- A simple parameterised Qiskit circuit that produces a single expectation value.
- A TorchQuantum kernel that can be used to compute a Gram matrix.
- Optional Gaussian shot noise to emulate a real backend.

The API mirrors the classical implementation so that it can be dropped
in without changing downstream code.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable, Sequence, Callable, Union
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Helper: default quantum circuit
# --------------------------------------------------------------------------- #
def _default_circuit() -> QuantumCircuit:
    """Return a minimal one‑qubit parameterised circuit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    return qc


# --------------------------------------------------------------------------- #
# TorchQuantum kernel ansatz
# --------------------------------------------------------------------------- #
class QKernelAnsatz(tq.QuantumModule):
    """Encode classical data through a list of quantum gates.

    The structure mirrors the reference QuantumKernelMethod implementation.
    """

    def __init__(self, func_list: Sequence[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QKernelAnsatz(
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


# --------------------------------------------------------------------------- #
# Unified hybrid layer
# --------------------------------------------------------------------------- #
class UnifiedHybridLayer:
    """Hybrid layer that can run a quantum circuit, a quantum kernel, or a classical model.

    Parameters
    ----------
    quantum_circuit : QuantumCircuit | None
        A parameterised Qiskit circuit.  If supplied, the class will use it
        to compute expectation values.  If ``None``, a simple default circuit
        is used.
    kernel : tq.QuantumModule | None
        A TorchQuantum kernel that can be evaluated as ``kernel(x, y)``.
    classical_model : nn.Module | None
        Optional classical neural network that consumes the output of the
        quantum backend.
    noise_shots : int | None
        When supplied, Gaussian noise with variance ``1/shots`` is added to
        deterministic outputs to emulate shot noise.
    noise_seed : int | None
        Seed for the random number generator used by the noise model.
    """

    def __init__(
        self,
        *,
        quantum_circuit: "QuantumCircuit | None" = None,
        kernel: "tq.QuantumModule | None" = None,
        classical_model: "nn.Module | None" = None,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.quantum_circuit = quantum_circuit or _default_circuit()
        self.kernel = kernel
        self.classical_model = classical_model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        if self.noise_shots is not None:
            self._rng = np.random.default_rng(self.noise_seed)

    # ------------------------------------------------------------------ #
    # Quantum evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_quantum(
        self,
        observables: Iterable["SparsePauliOp"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate a Qiskit circuit for a batch of parameter sets."""
        backend = Aer.get_backend("qasm_simulator")
        shots = self.noise_shots or 1024
        results: List[List[complex]] = []

        for values in parameter_sets:
            mapping = dict(zip(self.quantum_circuit.parameters, values))
            bound = self.quantum_circuit.assign_parameters(mapping, inplace=False)
            job = execute(bound, backend, shots=shots)
            result = job.result()
            counts = result.get_counts(bound)
            probs = np.array([v / shots for v in counts.values()])
            states = np.array([int(k, 2) for k in counts.keys()])
            expectation = np.sum(states * probs)

            # If a classical model is attached, feed the expectation through it
            if self.classical_model is not None:
                torch_input = torch.tensor([[expectation]], dtype=torch.float32)
                with torch.no_grad():
                    torch_output = self.classical_model(torch_input)
                expectation = torch_output.item()

            results.append([complex(expectation)])

        if self.noise_shots is not None:
            noisy = []
            for row in results:
                noisy_row = [
                    complex(
                        self._rng.normal(val.real, max(1e-6, 1 / self.noise_shots))
                        + 1j
                        * self._rng.normal(val.imag, max(1e-6, 1 / self.noise_shots))
                    )
                    for val in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results

    # ------------------------------------------------------------------ #
    # Kernel evaluation
    # ------------------------------------------------------------------ #
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute a Gram matrix using the wrapped TorchQuantum kernel."""
        if self.kernel is None:
            raise RuntimeError("No kernel configured.")
        matrix = np.array([[self.kernel(x, y).item() for y in b] for x in a])
        if self.noise_shots is not None:
            matrix = matrix + self._rng.normal(0, max(1e-6, 1 / self.noise_shots), matrix.shape)
        return matrix

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable["SparsePauliOp"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the hybrid model using the quantum backend."""
        return self._evaluate_quantum(observables, parameter_sets)

    def run(self, parameters: Sequence[float]) -> np.ndarray:
        """Convenience wrapper that evaluates a single set of parameters."""
        return np.array(self.evaluate([SparsePauliOp.from_list([("Y", 1)])], [parameters]))

__all__ = ["UnifiedHybridLayer"]
