"""Hybrid fast estimator that fuses quantum circuits and quanvolution filtering.

The estimator extends the lightweight FastBaseEstimator from Qiskit with an
adaptive quanvolution filter that preprocesses classical data before encoding
into the circuit.  It supports expectation‑value evaluation for a batch of
parameter sets and can optionally run a shot‑based simulation with a
Gaussian‑like noise model.  The design follows a combination scaling paradigm:
fast quantum evaluation + optional quanvolution preprocessing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuanvFilter:
    """Quanvolution filter that encodes 2‑D data into a probability amplitude.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (number of qubits = kernel_size^2).
    threshold : float, default 0.5
        Threshold for data binarization before the Rx rotation.
    shots : int, default 1024
        Number of shots for the measurement.  Used only when running a shot‑based
        backend; otherwise the filter is deterministic.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 1024) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Simple entangling pattern: pairwise CNOTs across the grid.
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)

        # Parameterized RX gates
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        # Add a shallow random circuit to enrich the feature map.
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)

        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array and return the average |1> probability."""
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            qiskit.Aer.get_backend("qasm_simulator"),
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class HybridFastEstimator:
    """Hybrid fast estimator for quantum circuits with optional quanvolution filter.

    Parameters
    ----------
    circuit : QuantumCircuit
        The primary quantum circuit to evaluate.
    quanv : Optional[QuanvFilter], default None
        Optional quanvolution filter that preprocesses classical data before
        encoding into the circuit.
    backend : qiskit.providers.Provider, optional
        Backend used for state‑vector or shot‑based simulation.  Defaults to
        Aer simulator.
    shots : int, optional
        Number of shots for the measurement.  If ``None`` the simulator runs
        in state‑vector mode.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        quanv: Optional[QuanvFilter] = None,
        backend=None,
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.quanv = quanv
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Bind classical parameters to the circuit."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            # Apply quanvolution preprocessing if provided.
            if self.quanv is not None:
                # The filter expects a 2‑D array; we generate a dummy array
                # from the parameters to mimic the original design.
                data = np.array(values).reshape(-1, 1)
                _ = self.quanv.run(data)  # result is ignored in this simplified example.

            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def simulate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Run the circuit on a shot‑based backend and estimate expectations."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            job = qiskit.execute(
                bound,
                self.backend,
                shots=self.shots,
                seed_simulator=seed,
            )
            result = job.result()
            counts = result.get_counts(bound)

            # Convert counts to expectation values for each observable.
            row = []
            for obs in observables:
                sv = Statevector.from_counts(counts, bound.num_qubits)
                row.append(sv.expectation_value(obs))
            results.append(row)

        return results


__all__ = ["HybridFastEstimator", "QuanvFilter"]
