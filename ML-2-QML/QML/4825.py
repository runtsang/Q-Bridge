"""Quantum backend for the hybrid fraud detection model.

Implements a parameterised circuit that takes 4 classical features as rotation
angles and applies a trainable variational block.  The circuit is evaluated
using Qiskit’s StatevectorSimulator or a real backend; the FastEstimator
wrapper adds optional shot noise.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal

class QuantumFraudLayer:
    """
    Parameterised quantum circuit for fraud detection.

    The circuit accepts 4 classical parameters (the outputs of the CNN)
    as rotation angles and then applies a 2‑local ansatz with trainable
    parameters.  The circuit outputs expectation values of Pauli‑Z on
    each qubit, which are returned as a 4‑dimensional tensor.
    """

    def __init__(self, n_wires: int = 4, reps: int = 2, seed: int | None = None):
        self.n_wires = n_wires
        self.reps = reps
        self.seed = seed or 42
        # Trainable parameters for the ansatz
        self.param_symbols = [
            f"theta_{i}_{j}" for i in range(reps) for j in range(n_wires)
        ]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        # Classical rotation angles as parameters
        cl_params = [Parameter(f"phi_{i}") for i in range(self.n_wires)]
        qc = QuantumCircuit(self.n_wires, self.n_wires)

        # Encode classical data
        for i in range(self.n_wires):
            qc.ry(cl_params[i], i)

        # Trainable variational block
        for rep in range(self.reps):
            for i in range(self.n_wires):
                qc.rx(Parameter(self.param_symbols[rep * self.n_wires + i]), i)
            for i in range(self.n_wires - 1):
                qc.cx(i, i + 1)

        # Measurement
        qc.measure_all()
        return qc

    def bind_and_execute(
        self,
        data: Sequence[float],
        shots: int | None = None,
        backend: str | None = None,
    ) -> List[float]:
        """
        Bind classical data to the circuit, execute, and return expectation values.

        Parameters
        ----------
        data: Sequence[float]
            4‑dimensional feature vector.
        shots: int | None
            If provided, run a shot‑based simulation; otherwise use
            StatevectorSimulator.
        backend: str | None
            Backend name for Qiskit (e.g., 'qasm_simulator').  Defaults to
           'statevector_simulator'.

        Returns
        -------
        List[float]
            Expectation values of Pauli‑Z for each qubit.
        """
        if len(data)!= self.n_wires:
            raise ValueError(f"Expected {self.n_wires} parameters, got {len(data)}")

        param_bind = {f"phi_{i}": data[i] for i in range(self.n_wires)}
        bound_qc = self.circuit.bind_parameters(param_bind)

        if shots is None:
            # Deterministic statevector evaluation
            backend = Aer.get_backend("statevector_simulator")
            transpiled = transpile(bound_qc, backend)
            result = backend.run(transpiled).result()
            state = Statevector(result.get_statevector())
            return [state.expectation_value(Operator("Z") ^ i) for i in range(self.n_wires)]

        # Shot‑based simulation
        backend = Aer.get_backend("qasm_simulator" if backend is None else backend)
        transpiled = transpile(bound_qc, backend)
        result = backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        probs = {int(k, 2): v / shots for k, v in counts.items()}
        # Compute expectation values from probabilities
        exp_vals = []
        for i in range(self.n_wires):
            exp = 0.0
            for bitstring, p in probs.items():
                bit = (bitstring >> i) & 1
                exp += p * (1 if bit == 0 else -1)
            exp_vals.append(exp)
        return exp_vals

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Vectorised call over a batch of feature vectors.

        Parameters
        ----------
        features: np.ndarray of shape (batch, 4)

        Returns
        -------
        np.ndarray of shape (batch, 4)
        """
        return np.array([self.bind_and_execute(f) for f in features])

class FastEstimator:
    """
    Lightweight estimator that evaluates a list of observables on a
    parameterised circuit and optionally adds Gaussian shot noise.
    """

    def __init__(self, circuit: QuantumCircuit, observables: Iterable[BaseOperator]):
        self.circuit = circuit
        self.observables = list(observables)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_qc = self._bind(values)
            if shots is None:
                backend = Aer.get_backend("statevector_simulator")
                result = backend.run(transpile(bound_qc, backend)).result()
                state = Statevector(result.get_statevector())
                row = [state.expectation_value(obs) for obs in self.observables]
            else:
                backend = Aer.get_backend("qasm_simulator")
                result = backend.run(transpile(bound_qc, backend), shots=shots).result()
                counts = result.get_counts()
                probs = {int(k, 2): v / shots for k, v in counts.items()}
                row = []
                for obs in self.observables:
                    # For simplicity, assume obs is PauliZ on a single qubit
                    exp = 0.0
                    for bitstring, p in probs.items():
                        bit = (bitstring & 1)  # only first qubit
                        exp += p * (1 if bit == 0 else -1)
                    row.append(exp)
            results.append(row)
        if seed is not None and shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy.append([complex(rng.normal(v.real, 1 / shots), v.imag) for v in row])
            return noisy
        return results

__all__ = ["QuantumFraudLayer", "FastEstimator"]
