"""Quantum‑centric implementation of the hybrid layer.

The class implements the same logical flow as the classical version but
uses a parameterised Qiskit circuit.  All parameters are exposed as
``qiskit.circuit.Parameter`` objects so that they can be optimised
through a variational routine.  The circuit is deliberately simple:
Hadamard gates, parameterised Ry/Rz rotations for the linear part,
separate RX rotations for the four LSTM‑style gates, a couple of
controlled‑RX gates to emulate the photonic fraud logic, and a final
measurement of all qubits.  The expectation value of the binary state
is returned as a single scalar, matching the output shape of the
classical implementation.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit

class UnifiedHybridQuantumLayer:
    """Quantum analogue of :class:`UnifiedHybridLayer`."""
    def __init__(self,
                 n_qubits: int = 2,
                 shots: int = 1024,
                 backend_name: str = "qasm_simulator") -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend(backend_name)

        # Parameter names in the order they will be bound
        self._param_names = [
            "w0", "w1", "b",
            "gf0", "gf1",      # forget gate
            "gi0", "gi1",      # input gate
            "gu0", "gu1",      # update gate
            "go0", "go1",      # output gate
            "f0", "f1",        # fraud logic
        ]
        self.params = {name: qiskit.circuit.Parameter(name) for name in self._param_names}

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # 1. Hadamard on all qubits
        self.circuit.h(range(self.n_qubits))

        # 2. Linear part
        self.circuit.ry(self.params["w0"], 0)
        self.circuit.ry(self.params["w1"], 1)
        self.circuit.rz(self.params["b"], 0)

        # 3. Forget gate
        self.circuit.rx(self.params["gf0"], 0)
        self.circuit.rx(self.params["gf1"], 1)

        # 4. Input gate
        self.circuit.rx(self.params["gi0"], 0)
        self.circuit.rx(self.params["gi1"], 1)

        # 5. Update gate
        self.circuit.rx(self.params["gu0"], 0)
        self.circuit.rx(self.params["gu1"], 1)

        # 6. Output gate
        self.circuit.rx(self.params["go0"], 0)
        self.circuit.rx(self.params["go1"], 1)

        # 7. Fraud logic (controlled‑RX)
        self.circuit.crx(self.params["f0"], 0, 1)
        self.circuit.crx(self.params["f1"], 1, 0)

        # 8. Measurement of all qubits
        self.circuit.measure_all()

    def run(self, param_values: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameter values.

        Parameters must be in the order of ``self._param_names``.
        """
        bind = {name: val for name, val in zip(self._param_names, param_values)}
        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = {state: count / self.shots for state, count in counts.items()}
        expectation = sum(int(state, 2) * prob for state, prob in probs.items())
        return np.array([expectation])

    def get_default_parameters(self) -> Tuple[float,...]:
        """Return a list of default parameter values that produce a neutral output."""
        return tuple(0.0 for _ in self._param_names)

def FCL() -> UnifiedHybridQuantumLayer:
    """Return the quantum layer class that mirrors the classical FCL API."""
    return UnifiedHybridQuantumLayer

__all__ = ["FCL", "UnifiedHybridQuantumLayer"]
