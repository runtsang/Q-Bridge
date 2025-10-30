"""Hybrid quantum estimator supporting Qiskit circuits, torchquantum modules,
and hybrid classical‑quantum pipelines.

The design follows a *combination* scaling paradigm: the estimator can
evaluate a pure quantum circuit, a pure torchquantum module, or a
combination where a classical preprocessing (e.g. ConvFilter or
SelfAttention) is applied before the quantum core.  Measurement
statistics are simulated with a configurable number of shots, and
optional Gaussian noise can be added to emulate shot‑noise in the
classical post‑processing stage.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable, Union
import torch
from torch import nn

# ----- Quantum base estimator -----
class QuantumBaseEstimator:
    """Evaluate expectation values of a Qiskit circuit or torchquantum module."""
    def __init__(self, model: Union[QuantumCircuit, Callable]) -> None:
        self.model = model
        if isinstance(model, QuantumCircuit):
            self.backend = Aer.get_backend("qasm_simulator")
            self.shots = 1024
        else:
            self.backend = None
            self.shots = None

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if isinstance(self.model, QuantumCircuit):
            return self._evaluate_circuit(observables, parameter_sets)
        else:
            return self.model.evaluate(observables, parameter_sets)

    def _evaluate_circuit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind_parameters(self.model, params)
            job = execute(circ, self.backend, shots=self.shots)
            counts = job.result().get_counts(circ)
            state = Statevector.from_counts(counts, circ.num_qubits)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    @staticmethod
    def _bind_parameters(circ: QuantumCircuit,
                         params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(circ.parameters):
            raise ValueError("Parameter mismatch.")
        mapping = {p: v for p, v in zip(circ.parameters, params)}
        return circ.assign_parameters(mapping, inplace=False)

# ----- Quantum estimator with shot noise -----
class Estimator(QuantumBaseEstimator):
    """Wrap a quantum estimator with Gaussian shot‑noise."""
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
            noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                                 rng.normal(val.imag, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

# ----- Classical‑to‑quantum hybrid estimator -----
class HybridEstimator:
    """
    Evaluate a hybrid pipeline where a classical preprocessing step (e.g.
    ConvFilter or SelfAttention) feeds data into a quantum core.
    """
    def __init__(
        self,
        quantum_core: Union[QuantumCircuit, Callable],
        *,
        preprocessor: Callable | None = None,
        noise_shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.quantum_core = quantum_core
        self.preprocessor = preprocessor
        self.noise_shots = noise_shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        # Pre‑process raw parameters
        if self.preprocessor is not None:
            parameter_sets = [self.preprocessor.run(np.array(params)) for params in parameter_sets]
        # Dispatch to quantum core
        quantum_est = Estimator(self.quantum_core,
                                shots=self.noise_shots,
                                seed=self.seed)
        raw = quantum_est.evaluate(observables, parameter_sets)
        # If additional classical post‑processing is needed, insert here.
        return raw

# ----- Quantum Self‑Attention circuit -----
def QuantumSelfAttention(n_qubits: int = 4) -> Callable:
    """Return a callable that builds and runs a self‑attention style circuit."""
    def _build_circuit(rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    return _build_circuit

# ----- Quantum Conv filter circuit -----
def QuantumConvFilter(kernel_size: int = 2,
                      threshold: float = 0.5,
                      shots: int = 100) -> Callable:
    """Return a callable that runs a quanvolution‑style circuit on a 2D patch."""
    n_qubits = kernel_size ** 2
    base_circ = QuantumCircuit(n_qubits)
    theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        base_circ.rx(theta[i], i)
    base_circ.barrier()
    base_circ += qiskit.circuit.random.random_circuit(n_qubits, 2)
    base_circ.measure_all()

    def _run(data: np.ndarray) -> float:
        data = np.reshape(data, (1, n_qubits))
        counts = 0
        for row in data:
            binding = {theta[i]: np.pi if val > threshold else 0 for i, val in enumerate(row)}
            job = execute(base_circ, Aer.get_backend("qasm_simulator"),
                          shots=shots, parameter_binds=[binding])
            result = job.result().get_counts(base_circ)
            for key, val in result.items():
                counts += sum(int(b) for b in key) * val
        return counts / (shots * n_qubits)

    return _run

__all__ = ["QuantumBaseEstimator", "Estimator", "HybridEstimator",
           "QuantumSelfAttention", "QuantumConvFilter"]
