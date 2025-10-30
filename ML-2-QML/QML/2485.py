"""Hybrid estimator that unifies quantum circuit evaluation with optional auto‑encoder and shot‑noise simulation."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler, SamplerResult
from qiskit_machine_learning.neural_networks import SamplerQNN


class HybridEstimator:
    """Evaluate a parametrised quantum circuit, optionally wrapped by an auto‑encoder circuit."""
    def __init__(self, circuit: QuantumCircuit, autoencoder: QuantumCircuit | None = None) -> None:
        self._circuit = circuit
        self._autoencoder = autoencoder
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        bound = self._circuit.assign_parameters(mapping, inplace=False)
        if self._autoencoder is not None:
            bound.compose(self._autoencoder, inplace=True)
        return bound

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        If `shots` is supplied, a :class:`qiskit.primitives.Sampler` is used to obtain noisy
        expectation values; otherwise a state‑vector simulation is performed.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circuit = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                sampler = Sampler(seed=seed)
                result: SamplerResult = sampler.run(circuit, shots=shots)
                counts = result.get_counts()
                exp = 0.0
                for outcome, count in counts.items():
                    exp += ((-1) ** int(outcome)) * count
                exp /= shots
                row = [exp for _ in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder utilities – borrowed from the QML seed
# --------------------------------------------------------------------------- #

def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a quantum auto‑encoder circuit with a swap‑test style read‑out."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])

    return circuit


def domain_wall_circuit(num_qubits: int, start: int, end: int) -> QuantumCircuit:
    """Apply X gates to a contiguous block of qubits."""
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc


def AutoencoderQNN(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a SamplerQNN that implements the auto‑encoder circuit."""
    ae_circ = auto_encoder_circuit(num_latent, num_trash)

    qnn = SamplerQNN(
        circuit=ae_circ,
        input_params=[],
        weight_params=ae_circ.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=Sampler(),
    )
    return qnn


__all__ = ["HybridEstimator", "auto_encoder_circuit", "domain_wall_circuit", "AutoencoderQNN"]
