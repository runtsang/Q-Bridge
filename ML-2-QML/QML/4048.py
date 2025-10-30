"""Hybrid quantum estimator that can evaluate a parametric circuit with optional
quantum autoencoder or fully connected layer.

The estimator supports deterministic evaluation via state‑vector simulation and
shot‑based Gaussian noise to emulate sampling uncertainty.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.quantum_info import BaseOperator, Statevector, Pauli
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RealAmplitudes
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import QuantumRegister, ClassicalRegister

# Ensure a deterministic simulator
algorithm_globals.random_seed = 42
backend = AerSimulator()

class FastBaseEstimator:
    """Hybrid quantum estimator.

    Parameters
    ----------
    circuit
        The primary parametric quantum circuit.
    autoencoder
        Optional quantum autoencoder (SamplerQNN) that maps raw parameters to a latent
        representation.
    fcl
        Optional quantum circuit that acts as a fully connected layer on the
        parameters before the main circuit.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        autoencoder: SamplerQNN | None = None,
        fcl: QuantumCircuit | None = None,
    ) -> None:
        self.circuit = circuit
        self.autoencoder = autoencoder
        self.fcl = fcl
        self._parameters = list(circuit.parameters)

    def _preprocess(self, param_values: Sequence[float]) -> Sequence[float]:
        """Apply optional FCL and autoencoder to the input parameters."""
        # FCL: evaluate the auxiliary circuit to obtain a single expectation value
        if self.fcl is not None:
            fcl_params = {self.fcl.parameters[0]: param_values[0]}
            bound_fcl = self.fcl.assign_parameters(fcl_params, inplace=False)
            state = Statevector.from_instruction(bound_fcl)
            # use expectation of Z on the only qubit as the new first parameter
            new_first = float(state.expectation_value(Pauli('Z')))
            param_values = [new_first] + list(param_values[1:])

        # Autoencoder: use the SamplerQNN to encode the parameters
        if self.autoencoder is not None:
            encoded = self.autoencoder.evaluate([param_values])[0]
            param_values = encoded
        return param_values

    def _bind(self, parameters: Sequence[float]) -> QuantumCircuit:
        if len(parameters)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameters))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Deterministic evaluation using state‑vector simulation."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            values = self._preprocess(values)
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate with optional shot noise using Gaussian perturbation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [float(rng.normal(complex(val).real, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def Autoencoder(
        num_latent: int,
        num_trash: int,
    ) -> SamplerQNN:
        """Return a quantum autoencoder based on a RealAmplitudes ansatz and a swap test."""
        def ansatz(num_qubits: int) -> QuantumCircuit:
            return RealAmplitudes(num_qubits, reps=5)

        def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)
            circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
            circuit.barrier()
            auxiliary_qubit = num_latent + 2 * num_trash
            circuit.h(auxiliary_qubit)
            for i in range(num_trash):
                circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
            circuit.h(auxiliary_qubit)
            circuit.measure(auxiliary_qubit, cr[0])
            return circuit

        circuit = auto_encoder_circuit(num_latent, num_trash)
        sampler = StatevectorSampler(backend=backend)
        return SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )

    @staticmethod
    def FCL() -> QuantumCircuit:
        """Return a simple parameterised quantum circuit mimicking a fully connected layer."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        return qc


__all__ = ["FastBaseEstimator"]
