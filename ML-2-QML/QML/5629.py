"""Quantum implementation of the fully‑connected layer.

The quantum version mirrors the classical API but replaces the
auto‑encoder and kernel with parameterised quantum circuits.
The sampler is a Qiskit SamplerQNN that outputs a 2‑class probability
distribution.

The module is intentionally lightweight – it can be run on a local
Aer simulator or any Qiskit backend that supports statevector or
sampling primitives.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit.quantum_info import Statevector


# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """
    Simple quantum kernel that encodes two input vectors into
    separate quantum states and returns the squared inner product.
    """
    def __init__(self, n_qubits: int, backend: AerSimulator | None = None) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator(method="statevector")

    def _statevector(self, vector: np.ndarray) -> Statevector:
        qc = QuantumCircuit(self.n_qubits)
        for i, angle in enumerate(vector):
            qc.ry(angle, i)
        return Statevector.from_instruction(qc)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        sv_x = self._statevector(x)
        sv_y = self._statevector(y)
        return float(abs(np.vdot(sv_x, sv_y)) ** 2)


# --------------------------------------------------------------------------- #
# Quantum sampler
# --------------------------------------------------------------------------- #
class SamplerQNN:
    """
    Quantum sampler that implements a simple 2‑qubit circuit with Ry
    rotations followed by a CX gate.  The measurement result is
    interpreted as a 2‑class probability via a classical softmax.
    """
    def __init__(self, backend: AerSimulator | None = None, shots: int = 1024) -> None:
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, 1)
        theta1 = Parameter("θ1")
        theta2 = Parameter("θ2")
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        qc.cx(0, 1)
        qc.measure(0, 0)
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit with the given parameters and return a softmaxed
        probability vector over the two measurement outcomes.
        """
        bound_qc = self.circuit.bind_parameters(
            {self.circuit.parameters[0]: params[0],
             self.circuit.parameters[1]: params[1]}
        )
        job = self.backend.run(bound_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_qc)
        probs = np.array([counts.get("0", 0), counts.get("1", 0)]) / self.shots
        # Softmax to keep the API consistent with the classical sampler
        exp = np.exp(probs)
        return exp / exp.sum()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.run(inputs)


# --------------------------------------------------------------------------- #
# Hybrid fully‑connected layer – quantum variant
# --------------------------------------------------------------------------- #
class FCL:
    """
    Quantum‑aware fully‑connected layer that mirrors the classical
    :class:`FCL` API.  It uses a quantum kernel to produce a scalar
    feature, a linear classical layer, and a quantum sampler to
    produce a 2‑class probability distribution.
    """
    def __init__(
        self,
        n_features: int = 1,
        mode: str = "quantum",
        *,
        reference_vectors: List[np.ndarray] | None = None,
        kernel_backend: AerSimulator | None = None,
        sampler_backend: AerSimulator | None = None,
    ) -> None:
        self.mode = mode.lower()
        if self.mode!= "quantum":
            raise ValueError("Quantum implementation only supports mode='quantum'")
        self.n_features = n_features

        # Reference vectors for kernel evaluation
        if reference_vectors is None:
            rng = np.random.default_rng(42)
            reference_vectors = [rng.normal(size=n_features) for _ in range(5)]
        self.references = reference_vectors

        self.kernel = QuantumKernel(n_features, backend=kernel_backend)
        self.linear = lambda x: np.tanh(x)  # simple scalar linear
        self.sampler = SamplerQNN(backend=sampler_backend)

    def _process_quantum(self, x: np.ndarray) -> float:
        sims = np.array([self.kernel(x, r) for r in self.references])
        # Reduce to a single scalar via mean
        return sims.mean()

    def forward(self, inputs: Iterable[float]) -> np.ndarray:
        x = np.array(list(inputs), dtype=np.float64)
        if self.mode == "quantum":
            feature = self._process_quantum(x)
        else:
            raise RuntimeError("Only quantum mode is implemented in this file")
        lin_out = self.linear(np.array([feature]))
        prob = self.sampler.forward(lin_out)
        return prob

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Compatibility wrapper that mimics the original anchor API."""
        return self.forward(thetas)


__all__ = ["FCL", "QuantumKernel", "SamplerQNN"]
