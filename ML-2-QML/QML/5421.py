"""
FastBaseEstimatorGen313 – quantum side
======================================

This module implements a variational quantum kernel and a lightweight
quantum estimator that can be used by the classical hybrid estimator.
The kernel is based on TorchQuantum and can be imported by the
``FastBaseEstimatorGen313`` class in the classical module.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Iterable, List, Sequence, Optional

# ---------------------------------------------------------------------------

class QuantumKernelModule(tq.QuantumModule):
    """Variational quantum kernel implemented with TorchQuantum.

    The ansatz consists of a single layer of Ry rotations on each qubit
    followed by a reverse application with negative parameters to encode
    the second data vector.  The kernel value is the absolute square of
    the overlap between the resulting states.

    Parameters
    ----------
    n_wires : int
        Number of qubits (features).  Default is 4.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Build the ansatz once
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the kernel matrix between two batches of data.

        Parameters
        ----------
        x, y : Tensor of shape (batch, n_wires)
            Data vectors to be encoded.
        """
        # Ensure correct shape
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)

        # Encode x
        self.q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Encode -y in reverse order
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Return kernel matrix (overlap magnitude)
        return torch.abs(self.q_device.states.view(-1)[0])

# ---------------------------------------------------------------------------

class FastBaseEstimatorGen313:
    """Quantum estimator that wraps a Qiskit circuit and provides
    expectation‑value evaluation with optional shot noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parametrised Qiskit circuit that implements a ``bind_parameters`` method.
    shots : int | None
        Number of shots to simulate.  If ``None`` the result is deterministic.
    """

    def __init__(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> None:
        self.circuit = circuit
        self.shots = shots

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for the given circuit and observables.

        The method follows the original QML FastBaseEstimator but adds
        optional shot‑noise simulation.  The circuit is run for each
        parameter set and the expectation value of each observable is
        returned.

        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_circ = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        # Add shot noise if requested
        if self.shots is not None:
            rng = np.random.default_rng(seed=42)
            for i, row in enumerate(results):
                noisy_row = [rng.normal(loc=val.real, scale=1.0 / self.shots) + 1j * rng.normal(loc=val.imag, scale=1.0 / self.shots)
                             for val in row]
                results[i] = noisy_row

        return results

__all__ = ["QuantumKernelModule", "FastBaseEstimatorGen313"]
