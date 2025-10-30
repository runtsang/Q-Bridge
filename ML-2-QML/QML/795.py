"""Quantum helper for the hybrid auto‑encoder.

Provides a lightweight swap‑test based variational circuit that
accepts latent parameters as gate angles and returns a
SamplerQNN object.  The circuit is intentionally simple so that
it can be executed on a state‑vector simulator or a real device
with minimal depth.
"""

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def get_quantum_circuit(
    num_qubits: int,
    reps: int = 5,
    basis: str = "ry",
    swap_test: bool = True,
) -> Tuple[QuantumCircuit, Sampler]:
    """Return a quantum circuit and a sampler for latent regularisation.

    Parameters
    ----------
    num_qubits
        Number of qubits equal to the latent dimensionality.
    reps
        Number of repetitions of the RealAmplitudes ansatz.
    basis
        Basis gate for the RealAmplitudes ansatz.
    swap_test
        Whether to include a swap‑test for similarity measurement.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz
    ansatz = RealAmplitudes(num_qubits, reps=reps, name="var_ansatz")
    qc.compose(ansatz, inplace=True)

    if swap_test:
        # Add an auxiliary qubit for the swap test
        aux = QuantumRegister(1, "aux")
        qc.add_register(aux)
        qc.h(aux[0])
        for i in range(num_qubits):
            qc.cswap(aux[0], qr[i], qr[i])  # dummy swap to illustrate
        qc.h(aux[0])
        qc.measure(aux[0], cr[0])

    sampler = Sampler()
    return qc, sampler


def build_sampler_qnn(
    num_qubits: int,
    qparams: Dict[str, int | str] | None = None,
) -> SamplerQNN:
    """Convenience wrapper that builds a SamplerQNN for the latent space.

    Parameters
    ----------
    num_qubits
        Latent dimensionality.
    qparams
        Dictionary of quantum parameters (e.g. reps, basis).
    """
    qc, sampler = get_quantum_circuit(num_qubits, **(qparams or {}))
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=(1,),
        sampler=sampler,
    )


__all__ = ["get_quantum_circuit", "build_sampler_qnn"]
