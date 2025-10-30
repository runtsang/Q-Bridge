"""Quantum‑centric auto‑encoder based on the original Qiskit seed.

The module offers a single :class:`AutoencoderHybrid` that builds a
swap‑test‑based variational circuit and exposes it as a
:class:`qiskit_machine_learning.neural_networks.SamplerQNN`.  The
class also provides a lightweight estimator that evaluates
expectation values of arbitrary :class:`BaseOperator`s.

Typical usage::
    from Autoencoder__gen231 import AutoencoderHybrid
    qmodel = AutoencoderHybrid(num_latent=3, num_trash=2)
    latent = qmodel.encode(input_vector)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Callable, Optional

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
#   Quantum auto‑encoder circuit builder – adapted from the seed          #
# --------------------------------------------------------------------------- #
def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build the swap‑test based quantum auto‑encoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=5),
        range(num_latent + num_trash),
        inplace=True,
    )
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    # swap test
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit

# --------------------------------------------------------------------------- #
#   Estimator – copy of the original FastBaseEstimator from the second seed #
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#   Hybrid class combining quantum auto‑encoder and estimator              #
# --------------------------------------------------------------------------- #
class AutoencoderHybrid:
    """
    Quantum auto‑encoder based on a swap‑test circuit and a
    :class:`~qiskit_machine_learning.neural_networks.SamplerQNN`.

    Parameters
    ----------
    num_latent : int
        Size of the latent register.
    num_trash : int, default 2
        Number of auxiliary qubits used for the swap test.
    seed : int, default 42
        Random seed used for the underlying Qiskit primitives.
    """

    def __init__(
        self,
        num_latent: int,
        *,
        num_trash: int = 2,
        seed: int = 42,
    ) -> None:
        algorithm_globals.random_seed = seed
        self._circuit = _auto_encoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=[],
            weight_params=self._circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )

    # ------------------------------------------------------------------ #
    #   Public API – quantum encoding and forward pass                  #
    # ------------------------------------------------------------------ #
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the underlying circuit for a given set of parameters.

        Parameters
        ----------
        inputs : np.ndarray
            Parameter vector of length ``len(self._circuit.parameters)``.
        Returns
        -------
        np.ndarray
            The probability distribution measured on the single output qubit.
        """
        bound = self._circuit.assign_parameters(dict(zip(self._circuit.parameters, inputs)), inplace=False)
        sampler = Sampler()
        result = sampler.run(bound, shots=1024).result()
        return np.array(result.quasi_dists[0])

    def quantum_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the :class:`SamplerQNN`.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector (ignored in this toy circuit but kept for API parity).
        Returns
        -------
        np.ndarray
            Output of the quantum neural network.
        """
        return self.qnn.forward(inputs).detach().numpy()

    # ------------------------------------------------------------------ #
    #   Estimator utility                                               #
    # ------------------------------------------------------------------ #
    def estimator(self, circuit: QuantumCircuit) -> FastBaseEstimator:
        """
        Return a :class:`FastBaseEstimator` that evaluates observables on the
        supplied circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parametrised circuit for which expectation values are required.
        """
        return FastBaseEstimator(circuit)

__all__ = [
    "AutoencoderHybrid",
    "FastBaseEstimator",
]
