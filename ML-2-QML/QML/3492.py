"""Hybrid quantum autoencoder using a swap‑test based VQE style circuit.

The quantum part mirrors the domain‑wall and swap‑test structure from
the original QML seed, but it is wrapped in a SamplerQNN so it can be
plugged into a hybrid training loop.  The circuit accepts a latent
vector as classical parameters, runs a RealAmplitudes ansatz, and
measures a swap‑test statistic that is interpreted as the reconstruction
quality.  The circuit is parameterised by a list of angles that can be
optimised jointly with a classical encoder/decoder if desired.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RGate, CXGate, HGate
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ----------------------------------------------------------------------
# 1. Configuration dataclass (identical to the classical counterpart)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for API parity)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# ----------------------------------------------------------------------
# 2. Helper: domain‑wall perturbation
# ----------------------------------------------------------------------
def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Insert an X‑string between start (inclusive) and end (exclusive)."""
    for q in range(start, end):
        circuit.x(q)
    return circuit

# ----------------------------------------------------------------------
# 3. Quantum autoencoder circuit
# ----------------------------------------------------------------------
def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build a swap‑test based quantum autoencoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the latent part with a variational ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, range(0, num_latent + num_trash))

    qc.barrier()

    # Auxiliary qubit for swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# ----------------------------------------------------------------------
# 4. SamplerQNN wrapper
# ----------------------------------------------------------------------
class HybridAutoencoder(SamplerQNN):
    """SamplerQNN that implements the swap‑test autoencoder."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        *,
        sampler: Sampler | None = None,
        weight_params: Sequence[str] | None = None,
    ) -> None:
        algorithm_globals.random_seed = 42
        sampler = sampler or Sampler()

        circuit = _auto_encoder_circuit(num_latent, num_trash)

        # Apply a domain‑wall to the first half of the trash qubits
        domain_wall_circuit = _domain_wall(QuantumCircuit(num_trash), 0, num_trash)
        circuit.compose(domain_wall_circuit, range(num_latent + num_trash), inplace=True)

        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=weight_params or [str(p) for p in circuit.parameters],
            interpret=self._interpret,
            output_shape=2,
            sampler=sampler,
        )

    @staticmethod
    def _interpret(output: np.ndarray) -> np.ndarray:
        """Return the raw measurement probabilities."""
        # The sampler returns a 2‑element histogram, we expose it directly.
        return output

# ----------------------------------------------------------------------
# 5. Convenience factory
# ----------------------------------------------------------------------
def HybridAutoencoderFactory(
    num_latent: int = 3,
    num_trash: int = 2,
    *,
    sampler: Sampler | None = None,
) -> HybridAutoencoder:
    """Return a ready‑to‑use sampler QNN."""
    return HybridAutoencoder(num_latent=num_latent, num_trash=num_trash, sampler=sampler)

__all__ = [
    "FraudLayerParameters",
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
]
