"""Quantum sampler that integrates a variational circuit with a quantum kernel.

The circuit encodes two input parameters via Ry gates followed by a
CX entanglement layer.  Weight parameters are applied in a second
entanglement block.  A StatevectorSampler is used to obtain measurement
probabilities.  A quantum kernel is defined as the squared overlap
between statevectors prepared from the input and a fixed set of
reference points; this kernel re‑weights the sampler output, providing
a data‑driven bias that complements the classical RBF kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler

class SamplerQNNGen065:
    """
    Hybrid quantum sampler that re‑weights circuit outcomes using a quantum
    kernel similarity to a set of reference points.
    """

    def __init__(
        self,
        reference_points: Sequence[Sequence[float]],
        gamma: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        reference_points
            Iterable of 2‑D points that form the kernel support set.
        gamma
            Kernel width hyper‑parameter for the quantum kernel.
        """
        self.reference_points = np.array(reference_points, dtype=np.float64)
        self.gamma = gamma
        self.sampler = StatevectorSampler()
        # Pre‑compute reference statevectors with zero weight parameters
        self.ref_states = [self._statevector_from_input(pt) for pt in self.reference_points]

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the parameterized circuit template."""
        qc = QuantumCircuit(2)
        # Input parameters (to be assigned later)
        qc.ry(0, 0)
        qc.ry(0, 1)
        qc.cx(0, 1)
        # Weight parameters
        qc.ry(0, 0)
        qc.ry(0, 1)
        qc.cx(0, 1)
        qc.ry(0, 0)
        qc.ry(0, 1)
        return qc

    def _statevector_from_input(self, x: Sequence[float]) -> Statevector:
        """Return the statevector for a given input, with zero weights."""
        qc = self._build_circuit()
        # Assign input parameters
        qc.assign_parameters({qc.parameters[0]: x[0], qc.parameters[1]: x[1]}, inplace=True)
        # Set weight parameters to zero
        for p in qc.parameters[2:]:
            qc.assign_parameters({p: 0.0}, inplace=True)
        return Statevector.from_instruction(qc)

    def kernel_similarity(self, x: Sequence[float]) -> np.ndarray:
        """Compute the quantum kernel similarity to all reference points."""
        sv_x = self._statevector_from_input(x)
        similarities = np.array([abs(sv_x.inner(ref))**2 for ref in self.ref_states])
        # Apply an RBF‑like scaling to the similarity
        return np.exp(-self.gamma * (1 - similarities))

    def sample(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        For each input, sample the circuit with parameters derived from the
        quantum kernel and return a probability distribution over the two
        basis states.
        """
        probs = []
        for x in inputs:
            qc = self._build_circuit()
            # Assign input parameters
            qc.assign_parameters({qc.parameters[0]: x[0], qc.parameters[1]: x[1]}, inplace=True)
            # Compute kernel similarity and map to weight parameters
            kernel_vals = self.kernel_similarity(x)
            # Use the first two kernel values as weight parameters
            for i, w in enumerate(kernel_vals[:2]):
                qc.assign_parameters({qc.parameters[2 + i]: w}, inplace=True)
            # Sample the circuit
            result = self.sampler.run(qc, shots=shots).result()
            counts = result.get_counts()
            # Convert counts to probability vector
            p0 = counts.get('00', 0) / shots
            p1 = counts.get('01', 0) / shots
            probs.append([p0, p1])
        return np.array(probs)

# Alias for backward compatibility
SamplerQNN = SamplerQNNGen065

__all__ = ["SamplerQNNGen065", "SamplerQNN"]
