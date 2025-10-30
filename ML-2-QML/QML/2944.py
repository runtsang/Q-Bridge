"""Hybrid estimator for Qiskit circuits with optional quantum auto‑encoding and shot noise.

The implementation mirrors the classical side but adds a quantum auto‑encoder
stage and supports both state‑vector and shot‑based evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes, SwapGate
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ConvertPauliMeasurementToStatevector

# --------------------------------------------------------------------------- #
# Helper: build a simple quantum auto‑encoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Construct a basic quantum auto‑encoder that compresses `num_latent + 2 * num_trash + 1`
    qubits into a latent subspace of size `num_latent`.

    The circuit uses a RealAmplitudes ansatz followed by a swap‑test that
    entangles the latent and trash qubits.  The final measurement of an auxiliary
    qubit yields the fidelity between the original and reconstructed states.
    """
    qr = QuantumCircuit(num_latent + 2 * num_trash + 1, name="autoencoder")
    # Ansatz on latent + first trash block
    qr.append(RealAmplitudes(num_latent + num_trash, reps=5), range(num_latent + num_trash))
    # Swap test
    aux = num_latent + 2 * num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure(aux, 0)  # measurement into classical register 0
    return qr

# --------------------------------------------------------------------------- #
# HybridEstimator class
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """
    Evaluate a parametrised quantum circuit (optionally preceded by a quantum
    auto‑encoder) over a batch of parameter sets and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The primary circuit to evaluate.
    autoencoder : Optional[QuantumCircuit]
        A pre‑built auto‑encoder circuit that will be composed before evaluation.
    shots : Optional[int]
        If provided, the evaluation will use the `Sampler` primitive with
        the given number of shots; otherwise a state‑vector evaluation is used.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        autoencoder: Optional[QuantumCircuit] = None,
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.autoencoder = autoencoder
        self.shots = shots
        self._sampler = Sampler() if shots is not None else None

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Bind a sequence of parameters to the circuit."""
        if len(param_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables
            A list of `BaseOperator` instances whose expectation values
            are computed for each parameter set.
        parameter_sets
            A sequence of parameter tuples to bind to the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)

            # Compose with auto‑encoder if supplied
            if self.autoencoder is not None:
                bound_circuit = bound_circuit.compose(
                    self.autoencoder, inplace=False
                )

            # Depending on the presence of shots, use state‑vector or sampler
            if self.shots is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # The sampler returns a list of samples; we compute expectation
                # values by averaging measurement results.
                result = self._sampler.run(
                    bound_circuit, shots=self.shots
                ).result()
                counts = result.get_counts()
                # Convert counts to probability distribution
                probs = {
                    int(k, 2): v / self.shots for k, v in counts.items()
                }
                # For each observable, compute its expectation value from the sampled distribution.
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, p in probs.items():
                        # Interpret bitstring as state vector index
                        exp += p * obs.data[bitstring, bitstring]
                    row.append(exp)
            results.append(row)

        return results


__all__ = ["HybridEstimator"]
