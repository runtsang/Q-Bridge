"""Quantum autoencoder implementation using Qiskit.

The class ``Autoencoder__gen299`` builds a depth‑constrained variational circuit.
It provides a feature‑encoding layer (``RawFeatureVector``) followed by a
``RealAmplitudes`` ansatz.  The circuit is wrapped into a ``SamplerQNN`` that
exposes a callable interface compatible with the classical training loop.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Hyper‑parameters
# --------------------------------------------------------------------------- #
MAX_DEPTH = 10  # upper bound for the ansatz depth


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #
class Autoencoder__gen299:
    """Depth‑constrained quantum autoencoder using a sampler QNN."""

    def __init__(
        self,
        num_qubits: int,
        *,
        depth: int = MAX_DEPTH,
        reps: int = 3,
        seed: int | None = None,
    ) -> None:
        algorithm_globals.random_seed = seed or 42
        self.num_qubits = num_qubits
        self.depth = depth
        self.reps = reps
        self.circuit = self._build_circuit()

    # --------------------------------------------------------------------- #
    # Circuit construction
    # --------------------------------------------------------------------- #
    def _build_circuit(self) -> QuantumCircuit:
        """Build a RealAmplitudes ansatz with a feature‑encoding layer."""
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        # Feature encoding
        feat = RawFeatureVector(self.num_qubits)
        qc.compose(feat, range(self.num_qubits), inplace=True)

        # Ansatz with depth control
        ansatz = RealAmplitudes(self.num_qubits, reps=self.reps)
        qc.compose(ansatz, range(self.num_qubits), inplace=True)

        # Swap‑test style measurement for a single qubit
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        return qc

    # --------------------------------------------------------------------- #
    # SamplerQNN wrapper
    # --------------------------------------------------------------------- #
    def get_qnn(self) -> SamplerQNN:
        """Return a SamplerQNN ready for integration with classical training."""
        sampler = Sampler()
        # The QNN has no trainable input_params; all parameters are circuit weights
        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=sampler,
        )
        return qnn

    # --------------------------------------------------------------------- #
    # Utility: depth estimator
    # --------------------------------------------------------------------- #
    def estimate_depth(self) -> int:
        """Return the number of two‑qubit gates in the circuit."""
        return len([g for g in self.circuit.data if g[0].name in {"cx", "cz", "swap"}])

    # --------------------------------------------------------------------- #
    # Representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"<Autoencoder__gen299 qubits={self.num_qubits} "
            f"depth={self.depth} reps={self.reps}>"
        )


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def Autoencoder(num_qubits: int, **kwargs) -> Autoencoder__gen299:
    """Return a ready‑to‑use quantum autoencoder instance."""
    return Autoencoder__gen299(num_qubits, **kwargs)


__all__ = ["Autoencoder", "Autoencoder__gen299"]
