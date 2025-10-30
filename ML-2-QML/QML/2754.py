"""Quantum‑classical hybrid classification – quantum side.

Provides a circuit factory and a lightweight wrapper that evaluates
the circuit on a Qiskit Aer simulator.  The wrapper exposes a
``run`` method that accepts a list of parameter values and returns
the expectation value of the Z operator for each qubit.  It also
stores the last inputs to support the parameter‑shift gradient
implemented in the classical module.

"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, assemble, transpile, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torch

# --------------------------------------------------------------------------- #
# Circuit factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Mirrors the classical network built in ``build_classical_classifier``.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    qc.measure_all()  # measurement for expectation calculation
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, encoding, weights, observables

# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """
    Thin wrapper that runs a Qiskit circuit on a simulator and returns
    the expectation value of the Z operator for each qubit.
    """
    def __init__(self, circuit: QuantumCircuit, backend, shots: int = 100):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.last_inputs = []

    def run(self, params: list[float]) -> list[float]:
        """
        Evaluate the circuit for a list of parameter values.

        Parameters
        ----------
        params : list[float]
            Parameter values for the encoding qubit(s).  The variational
            parameters are fixed to zero for each evaluation.

        Returns
        -------
        list[float]
            Expectation values of the Z operator for each qubit.
        """
        if not isinstance(params, list):
            params = params.tolist()
        self.last_inputs = params

        expect_vals = []
        for val in params:
            # Bind the encoding parameter to the current value, all weights to 0
            binds = {self.circuit.parameters[0]: val}
            for p in self.circuit.parameters[1:]:
                binds[p] = 0.0
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[binds])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Compute expectation of Z for the single qubit
            exp = 0.0
            for state, cnt in counts.items():
                bit = int(state[-1])  # single qubit
                exp += ((-1) ** bit) * cnt
            exp /= self.shots
            expect_vals.append(exp)
        return expect_vals

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
__all__ = ["build_classifier_circuit", "QuantumCircuitWrapper"]
