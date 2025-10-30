"""Hybrid estimator for quantum auto‑encoders.

This module implements the quantum counterpart to :class:`HybridAutoEncoderEstimator` from the
classical module.  It evaluates expectation values of a parametrised circuit and can train
a quantum auto‑encoder using a :class:`SamplerQNN` and the COBYLA optimiser.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper functions – quantum auto‑encoder construction
# --------------------------------------------------------------------------- #

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a parametrised auto‑encoder circuit with a swap‑test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz – RealAmplitudes on latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test auxiliary qubit
    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)

    # Measurement – keep only the auxiliary qubit
    circuit.measure(auxiliary, cr[0])
    return circuit


# --------------------------------------------------------------------------- #
# Hybrid estimator – quantum version
# --------------------------------------------------------------------------- #

class HybridAutoEncoderEstimator:
    """
    Quantum estimator that mirrors the classical :class:`HybridAutoEncoderEstimator`.
    Parameters
    ----------
    circuit : QuantumCircuit, optional
        Pre‑defined circuit to evaluate.  If ``None``, a default auto‑encoder
        circuit is constructed from ``num_latent`` and ``num_trash``.
    num_latent : int, optional
        Number of latent qubits for the auto‑encoder.  Required if ``circuit`` is
        ``None``.
    num_trash : int, optional
        Number of trash qubits for the auto‑encoder.  Required if ``circuit`` is
        ``None``.
    """
    def __init__(self, circuit: QuantumCircuit | None = None, *, num_latent: int | None = None, num_trash: int | None = None) -> None:
        if circuit is None:
            if num_latent is None or num_trash is None:
                raise ValueError("num_latent and num_trash must be provided when circuit is None.")
            circuit = _auto_encoder_circuit(num_latent, num_trash)
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    # ----------------------------------------------------------- #
    # Evaluation – copied and extended from the quantum FastBaseEstimator
    # ----------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values for each parameter set and observable.
        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators for which expectation values are requested.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains values for all circuit parameters.
        shots : int, optional
            If supplied, the expectation values are sampled with the given number of shots
            via :class:`StatevectorSampler`.  If ``None``, the exact expectation value
            is returned.
        Returns
        -------
        results : list[list[complex]]
            Outer list aligns with ``parameter_sets``, inner list with ``observables``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                sampler = Sampler()
                samples = sampler.run(bound_circ, shots=shots).result().samples
                # Convert classical measurement to expectation
                probs = {tuple(k): v for k, v in samples.items()}
                row = [self._sample_expectation(obs, probs) for obs in observables]
            results.append(row)
        return results

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    @staticmethod
    def _sample_expectation(obs: BaseOperator, probs: dict[tuple[int], float]) -> complex:
        """Compute expectation value from sampled probabilities."""
        # For simplicity, we assume obs is diagonal in the computational basis.
        # A full implementation would decompose obs into Pauli terms.
        expectation = 0.0
        for bitstring, p in probs.items():
            # interpret bitstring as integer
            val = 1.0 if sum(bitstring) % 2 == 0 else -1.0  # placeholder mapping
            expectation += val * p
        return expectation

    # ----------------------------------------------------------- #
    # Quantum auto‑encoder training – adapted from seed
    # ----------------------------------------------------------- #
    def train_autoencoder(
        self,
        data: List[QuantumCircuit],
        *,
        epochs: int = 50,
        shots: int = 1024,
    ) -> list[float]:
        """
        Train a quantum auto‑encoder using a :class:`SamplerQNN` and COBYLA.
        Parameters
        ----------
        data : list of QuantumCircuit
            Training dataset of circuits (or states) that the auto‑encoder should reconstruct.
        epochs : int
            Number of optimisation steps.
        shots : int
            Number of shots used to evaluate the loss at each step.
        Returns
        -------
        loss_history : list[float]
            Loss value after each optimisation step.
        """
        # Build a SamplerQNN that uses the underlying circuit as the weight circuit
        qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=[],
            weight_params=self._circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=Sampler(),
        )

        optimiser = COBYLA(maxiter=epochs)
        loss_history: list[float] = []

        for _ in range(epochs):
            # Sample current parameters
            current_params = np.random.uniform(-np.pi, np.pi, size=len(self._parameters))
            # Compute loss as mean squared error over the dataset
            loss = 0.0
            for circ in data:
                # Bind training circuit to current parameters
                bound = self._bind(current_params)
                bound.compose(circ, inplace=True)
                state = Statevector.from_instruction(bound)
                recon = qnn.forward(state.data.reshape(1, -1))
                loss += np.mean((recon - state.data) ** 2)
            loss /= len(data)
            loss_history.append(float(loss))
            # Update parameters
            current_params = optimiser.minimize(lambda p: loss, current_params)
        return loss_history

    # ----------------------------------------------------------- #
    # Convenience factory – mirrors the original Autoencoder helper
    # ----------------------------------------------------------- #
    @staticmethod
    def build_autoencoder(num_latent: int, num_trash: int) -> "HybridAutoEncoderEstimator":
        """Return an estimator wrapping the default auto‑encoder circuit."""
        return HybridAutoEncoderEstimator(circuit=_auto_encoder_circuit(num_latent, num_trash))


__all__ = ["HybridAutoEncoderEstimator"]
