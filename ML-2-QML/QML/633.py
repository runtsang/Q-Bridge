"""Variational quantum classifier with entanglement and gradient support."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence

import math
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """
    A variational quantum circuit that emulates a classical classifier
    while exposing quantum‑specific operations such as parameter‑shift
    gradients and flexible entanglement patterns.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int, default 2
        Number of variational layers.
    entanglement : str, default "full"
        Entanglement pattern: "full", "circular", or "none".
    encoding : str, default "rx"
        Feature encoding gate: "rx", "ry", or "rz".
    measurement : str, default "z"
        Observable to measure: "z" or "x".
    backend : str, default "statevector_simulator"
        Backend for state‑vector evaluation.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        entanglement: str = "full",
        encoding: str = "rx",
        measurement: str = "z",
        backend: str = "statevector_simulator",
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.measurement = measurement
        self.backend = Aer.get_backend(backend)

        # Build circuit
        self.circuit, self.feature_params, self.variational_params, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Sequence[ParameterVector], Sequence[ParameterVector], List[SparsePauliOp]]:
        """Internal helper to construct the variational circuit."""
        # Feature encoding
        feature_params = ParameterVector("x", self.num_qubits)
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            gate_name = {"rx": "rx", "ry": "ry", "rz": "rz"}[self.encoding]
            getattr(qc, gate_name)(feature_params[i], i)

        # Variational layers
        variational_params = ParameterVector("theta", self.num_qubits * self.depth)
        idx = 0
        for d in range(self.depth):
            # Single‑qubit rotations
            for q in range(self.num_qubits):
                rot_gate = qc.ry if d % 2 == 0 else qc.rz
                rot_gate(variational_params[idx], q)
                idx += 1
            # Entangling layer
            if self.entanglement == "full":
                for q in range(self.num_qubits):
                    qc.cz(q, (q + 1) % self.num_qubits)
            elif self.entanglement == "circular":
                for q in range(self.num_qubits - 1):
                    qc.cz(q, q + 1)
            elif self.entanglement == "none":
                pass

        # Observables
        if self.measurement == "z":
            observables = [SparsePauliOp(f"I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        elif self.measurement == "x":
            observables = [SparsePauliOp(f"I" * i + "X" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        else:
            raise ValueError(f"Unsupported measurement: {self.measurement}")

        return qc, feature_params, variational_params, observables

    def expectation(self, feature_vector: Sequence[float], variational_vector: Sequence[float]) -> List[float]:
        """
        Compute expectation values of the observables for a given input.

        Parameters
        ----------
        feature_vector : Sequence[float]
            Classical data to encode.
        variational_vector : Sequence[float]
            Current values of the variational parameters.

        Returns
        -------
        List[float]
            Expectation value for each observable.
        """
        param_map = {**{p: float(val) for p, val in zip(self.feature_params, feature_vector)},
                     **{p: float(val) for p, val in zip(self.variational_params, variational_vector)}}
        bound_qc = self.circuit.bind_parameters(param_map)
        result = execute(bound_qc, self.backend, shots=1).result()
        state = result.get_statevector()
        expectations = [op.expectation_value(state).real for op in self.observables]
        return expectations

    def parameter_shift_gradient(self, feature_vector: Sequence[float], variational_vector: Sequence[float]) -> List[float]:
        """
        Estimate the gradient of the expectation value w.r.t. each variational parameter
        using the parameter‑shift rule on a state‑vector backend.

        Parameters
        ----------
        feature_vector : Sequence[float]
            Classical data to encode.
        variational_vector : Sequence[float]
            Current values of the variational parameters.

        Returns
        -------
        List[float]
            Gradient vector of length ``len(variational_params)``.
        """
        shift = math.pi / 2
        grads = []

        for i, param in enumerate(self.variational_params):
            plus_vec = list(variational_vector)
            minus_vec = list(variational_vector)
            plus_vec[i] += shift
            minus_vec[i] -= shift
            exp_plus = self.expectation(feature_vector, plus_vec)[i]
            exp_minus = self.expectation(feature_vector, minus_vec)[i]
            grads.append((exp_plus - exp_minus) / (2 * math.sin(shift)))
        return grads

__all__ = ["QuantumClassifierModel"]
