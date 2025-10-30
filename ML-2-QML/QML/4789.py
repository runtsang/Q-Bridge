"""Quantum primitives that complement the hybrid quanvolution autoencoder."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

# --- Quantum autoencoder circuit -------------------------------------------
def quantum_autoencoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """
    Build a quantum autoencoder using a RealAmplitudes ansatz followed by a
    swap‑test that projects onto the latent subspace.
    """
    n_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary for swap test
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode data into the first ``num_latent + num_trash`` qubits
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=reps), range(num_latent + num_trash), inplace=True)
    qc.barrier()

    aux = num_latent + 2 * num_trash
    qc.h(aux)  # Prepare auxiliary qubit for swap test
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --- Sampler‑based QNN ------------------------------------------------------
def build_qnn(num_latent: int, num_trash: int, seed: int | None = None) -> SamplerQNN:
    """
    Return a SamplerQNN that can be trained with classical optimizers.
    """
    algorithm_globals.random_seed = seed or 42
    qc = quantum_autoencoder_circuit(num_latent, num_trash)
    # All parameters are trainable; no input parameters for simplicity.
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=1,
        sampler=StatevectorSampler(),
    )
    return qnn

# --- FastBaseEstimator (quantum) -------------------------------------------
class FastBaseEstimator:
    """Evaluates expectation values of a parametrised circuit on a statevector."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables,
                 param_sets: Sequence[Sequence[float]]) -> list[list[complex]]:
        """Return expectation values for each observable and parameter set."""
        results: list[list[complex]] = []
        for vals in param_sets:
            circ = self._bind(vals)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["quantum_autoencoder_circuit", "build_qnn", "FastBaseEstimator"]
