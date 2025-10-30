"""Quantum helper for the hybrid autoencoder.

Provides a quantum circuit builder and a subclass of the classical
HybridAutoencoder that attaches a SamplerQNN for latent refinement.
"""

from __future__ import annotations

import Autoencoder__gen449 as ml_module
BaseHybrid = ml_module.QuantumAutoencoderHybrid

from typing import List

import torch

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
#  Quantum circuit construction
# --------------------------------------------------------------------------- #

def build_quantum_autoencoder(num_latent: int,
                              num_trash: int = 2,
                              reps: int = 5) -> QuantumCircuit:
    """Return a quantum circuit implementing a swap‑test autoencoder.

    The circuit encodes the latent vector into the first ``num_latent`` qubits,
    uses ``num_trash`` auxiliary qubits, and performs a swap‑test on a
    single ancilla to compute the fidelity with the original state.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode the latent vector into the first ``num_latent`` qubits.
    circuit.compose(RealAmplitudes(num_latent + num_trash,
                                   reps=reps),
                    range(0, num_latent + num_trash),
                    inplace=True)

    # Swap‑test
    ancilla = num_latent + 2 * num_trash
    circuit.h(ancilla)
    for i in range(num_trash):
        circuit.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
    circuit.h(ancilla)
    circuit.measure(ancilla, cr[0])

    return circuit

# --------------------------------------------------------------------------- #
#  SamplerQNN wrapper
# --------------------------------------------------------------------------- #

def build_sampler_qnn(circuit: QuantumCircuit,
                      weight_params: List[object],
                      seed: int | None = None) -> SamplerQNN:
    """Instantiate a SamplerQNN for the given circuit."""
    if seed is not None:
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

    # The circuit has no input parameters, only weights.
    # We use the identity interpretation to obtain raw probabilities.
    return SamplerQNN(circuit=circuit,
                      input_params=[],
                      weight_params=weight_params,
                      interpret=lambda x: x,
                      output_shape=2,
                      sampler=StatevectorSampler())

# --------------------------------------------------------------------------- #
#  Hybrid subclass
# --------------------------------------------------------------------------- #

class QuantumAutoencoderHybrid(BaseHybrid):
    """Hybrid class that attaches a quantum refinement step to the classical model."""
    def __init__(self, cfg: ml_module.QuantumAutoencoderHybridConfig) -> None:
        super().__init__(cfg)

        # Build the quantum circuit and SamplerQNN
        self.quantum_circuit = build_quantum_autoencoder(cfg.latent_dim,
                                                        cfg.num_trash,
                                                        cfg.ansatz_reps)
        # The circuit parameters are the weights of the RealAmplitudes ansatz.
        weight_params = list(self.quantum_circuit.parameters)
        self.qnn = build_sampler_qnn(self.quantum_circuit,
                                     weight_params,
                                     seed=42)

    # The forward method is inherited from BaseHybrid; it will call refine_quantum.
    # No further changes are necessary.

__all__ = ["QuantumAutoencoderHybrid"]
