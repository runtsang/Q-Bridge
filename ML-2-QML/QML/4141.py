from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1. Photonic layer parameters
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

def _apply_photonic_layer(q, params: FraudLayerParameters, clip: bool = True):
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(_clip(r, 5) if clip else r, phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(_clip(r, 5) if clip else r, phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(_clip(k, 1) if clip else k) | q[i]

def build_photonic_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_photonic_layer(q, layer, clip=True)
    return prog

# --------------------------------------------------------------------------- #
# 2. Quantum hybrid auto‑encoder
# --------------------------------------------------------------------------- #

class HybridAutoencoder:
    """A quantum auto‑encoder that combines a swap‑test based circuit with optional
    photonic‑inspired parameterised layers and exposes a FastBaseEstimator‑style
    evaluation interface."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        num_trash: int,
        photonic_params: Sequence[FraudLayerParameters] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.photonic_params = photonic_params

        self.circuit = self._build_circuit()
        self.sampler = Sampler()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Optional photonic layer encoded as parameterised rotations
        if self.photonic_params:
            for idx, params in enumerate(self.photonic_params):
                theta = Parameter(f"θ_{idx}")
                phi = Parameter(f"φ_{idx}")
                circuit.ry(theta, qr[0])
                circuit.rz(phi, qr[0])

        # Ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=5)
        circuit.append(ansatz, range(0, self.latent_dim + self.num_trash))

        circuit.barrier()

        # Swap test
        aux = self.latent_dim + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values of *observables* for each parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = {p: val for p, val in zip(self.circuit.parameters, params)}
            bound_circ = self.circuit.assign_parameters(bound, inplace=False)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but uses a state‑vector sampler and adds Gaussian
        noise to emulate finite‑shot statistics."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = {p: val for p, val in zip(self.circuit.parameters, params)}
            bound_circ = self.circuit.assign_parameters(bound, inplace=False)
            sample = self.sampler.run(bound_circ, shots=shots).result()
            probs = sample.get_counts(bound_circ).values()
            # Convert probabilities to state‑vector amplitudes
            amplitudes = np.sqrt(np.array(list(probs)) / shots)
            state = Statevector(amplitudes)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if seed is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row]
                noisy.append(noisy_row)
            return noisy

        return results

__all__ = ["HybridAutoencoder", "FraudLayerParameters", "build_photonic_program"]
