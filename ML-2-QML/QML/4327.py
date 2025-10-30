"""Hybrid estimator for quantum models using Qiskit.

The class evaluates a parametrised quantum circuit (or list of circuits)
for a set of parameter values and returns expectation values of a set
of observables.  Shot noise can be simulated by specifying a number of
shots.  Helper functions build common quantum primitives such as a
quanvolution filter, a simple convolution filter implemented with a
classical circuit, and a fully‑connected quantum layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator

BaseOperator = Operator


class HybridEstimator:
    """Evaluate a Qiskit quantum circuit for batches of parameters and observables.

    Parameters
    ----------
    circuit
        A single ``QuantumCircuit`` or a list of circuits that will be
        executed sequentially.  Each circuit must be parametrised with
        ``Parameter`` objects.
    shots
        Number of shots for measurement statistics.  ``None`` runs a
        state‑vector simulation and returns exact expectation values.
    seed
        Random seed for reproducibility of the shot noise.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Sequence[QuantumCircuit]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(circuit, QuantumCircuit):
            self.circuits = [circuit]
        else:
            self.circuits = list(circuit)
        self.shots = shots
        self.backend = AerSimulator()
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        Each row corresponds to a parameter set and each column to an
        observable.  If ``self.shots`` is set, measurement noise is added
        to the expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Bind parameters to all circuits
            bound_circuits = []
            for circ in self.circuits:
                if len(params)!= len(circ.parameters):
                    raise ValueError("Parameter count mismatch for bound circuit.")
                mapping = dict(zip(circ.parameters, params))
                bound_circuits.append(circ.assign_parameters(mapping))

            # Execute the circuit(s)
            if self.shots is None:
                # State‑vector simulation
                state = Statevector.from_instruction(bound_circuits[0])
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound_circuits[0],
                    self.backend,
                    shots=self.shots,
                )
                result = job.result()
                counts = result.get_counts(bound_circuits[0])
                # Compute expectation values from counts
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, cnt in counts.items():
                        # Convert bitstring to integer
                        val = int(bitstring, 2)
                        exp += val * cnt
                    exp /= self.shots
                    row.append(complex(exp))

                # Add shot noise
                row = [
                    val + self.rng.normal(0, 1 / self.shots)
                    for val in row
                ]

            results.append(row)

        return results


# --------------------------------------------------------------------------- #
# Quantum primitives
# --------------------------------------------------------------------------- #

def create_quanvolution_filter() -> QuantumCircuit:
    """Return a 2‑qubit quantum filter that acts on 2×2 image patches."""
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += qiskit.circuit.random.random_circuit(n_qubits, 2)
    qc.measure_all()
    return qc


def create_conv_filter() -> QuantumCircuit:
    """A toy classical filter implemented as a quantum circuit."""
    # For demonstration, use a single qubit with a parameterised rotation.
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.h(0)
    qc.ry(theta, 0)
    qc.measure_all()
    return qc


def create_fcl() -> QuantumCircuit:
    """Fully‑connected quantum layer implemented with a single qubit."""
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.h(0)
    qc.ry(theta, 0)
    qc.measure_all()
    return qc


__all__ = [
    "HybridEstimator",
    "create_quanvolution_filter",
    "create_conv_filter",
    "create_fcl",
]
