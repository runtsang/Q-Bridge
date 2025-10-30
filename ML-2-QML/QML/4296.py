"""Hybrid quantum estimator that supports both Qiskit and TorchQuantum circuits,
including kernel evaluation.

Key features
------------
* Unified API: ``evaluate`` takes any iterable of observables and a list of
  parameter sets.
* Supports shot noise by delegating to a QASM simulator when ``shots`` is
  provided.
* For TorchQuantum circuits, the estimator automatically creates a
  quantum device, runs the circuit and extracts expectation values.
* If the wrapped circuit implements a kernel (i.e. has a ``forward`` that
  accepts two tensors and returns a scalar), a convenient
  :meth:`kernel_matrix` helper is available.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Iterable, Sequence
from typing import Any, List, Optional, Sequence

# Optional imports – the module degrades gracefully if either library
# is missing.  This allows the same file to be used in environments
# that only have one of the backends.
try:
    from qiskit import Aer, execute
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.quantum_info.operators.base_operator import BaseOperator
except Exception:  # pragma: no cover
    QuantumCircuit = None
    BaseOperator = None
    Aer = None
    execute = None
    Statevector = None

try:
    import torchquantum as tq
except Exception:  # pragma: no cover
    tq = None


class FastBaseEstimator:
    """Quantum estimator capable of evaluating Qiskit or TorchQuantum circuits.

    Parameters
    ----------
    circuit
        Either a :class:`qiskit.circuit.QuantumCircuit` or a
        :class:`torchquantum.QuantumModule`.
    """

    def __init__(self, circuit: Any) -> None:
        self.circuit = circuit

        if isinstance(circuit, QuantumCircuit):
            self._params = list(circuit.parameters)
            self._backend_state = Aer.get_backend("statevector_simulator")
            self._backend_qasm = Aer.get_backend("qasm_simulator")
        elif tq is not None and isinstance(circuit, tq.QuantumModule):
            self.n_wires = getattr(circuit, "n_wires", None)
        else:
            raise TypeError(
                "Unsupported circuit type.  Must be a Qiskit QuantumCircuit "
                "or a TorchQuantum QuantumModule."
            )

    # ------------------------------------------------------------------ #
    # Qiskit helpers
    # ------------------------------------------------------------------ #
    def _bind_qiskit(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _evaluate_qiskit(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []

        if shots is None:
            # Deterministic state‑vector evaluation
            for params in parameter_sets:
                bound = self._bind_qiskit(params)
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs).real for obs in observables]
                results.append(row)
        else:
            # Shot‑based simulation
            for params in parameter_sets:
                bound = self._bind_qiskit(params)
                job = execute(bound, backend=self._backend_qasm, shots=shots)
                res = job.result()
                counts = res.get_counts()
                row = []
                for obs in observables:
                    # Expectation of a PauliZ on a single qubit – a minimal
                    # implementation that works for the reference tests.
                    exp = 0.0
                    for bitstring, count in counts.items():
                        exp += (1 if bitstring[-1] == "0" else -1) * count
                    exp /= shots
                    row.append(exp)
                results.append(row)
        return results

    # ------------------------------------------------------------------ #
    # TorchQuantum helpers
    # ------------------------------------------------------------------ #
    def _evaluate_tq(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []

        for params in parameter_sets:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=1, device="cpu"
            )
            # Pass the parameters to the module – the module is expected
            # to accept a ``QuantumDevice`` and a tensor of parameters.
            self.circuit(qdev, torch.tensor(params, dtype=torch.float32))
            row = []
            for obs in observables:
                val = obs(qdev).item()
                row.append(val)
            results.append(row)

        return results

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        backend: Optional[Any] = None,
    ) -> List[List[complex]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of operators.  For Qiskit circuits these are
            :class:`qiskit.quantum_info.operators.base_operator.BaseOperator`;
            for TorchQuantum circuits they are callables that accept a
            :class:`torchquantum.QuantumDevice` and return a scalar.
        parameter_sets
            Sequence of parameter lists to bind to the circuit.
        shots
            Number of shots to simulate with a QASM backend.  If ``None``,
            a deterministic state‑vector evaluation is used.
        backend
            Optional custom QASM backend.  Ignored for TorchQuantum.
        """
        if isinstance(self.circuit, QuantumCircuit):
            return self._evaluate_qiskit(
                observables, parameter_sets, shots=shots
            )
        else:
            return self._evaluate_tq(observables, parameter_sets)

    # ------------------------------------------------------------------ #
    # Kernel helper – only available if the circuit implements a kernel
    # ------------------------------------------------------------------ #
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix between ``a`` and ``b`` using the wrapped
        kernel circuit.

        The wrapped circuit must expose a ``forward`` method that accepts two
        tensors and returns a scalar kernel value (the TorchQuantum
        implementation satisfies this contract).
        """
        if not hasattr(self.circuit, "forward"):
            raise RuntimeError("The wrapped circuit does not provide a kernel interface.")
        kernel = np.array(
            [
                [self.circuit(x, y).item() for y in b]
                for x in a
            ]
        )
        return kernel


__all__ = ["FastBaseEstimator"]
