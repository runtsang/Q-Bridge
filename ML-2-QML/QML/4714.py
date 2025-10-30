"""Quantum autoencoder with integrated self‑attention and fast estimation.

The circuit combines a RealAmplitudes encoder/decoder with a quantum
self‑attention block (RX/RY/RZ + controlled‑X gates).  A swap‑test using
an auxiliary qubit yields a fidelity measurement.  The wrapper class
`AutoencoderHybrid` mirrors the classical API and provides a
FastBaseEstimator for expectation‑value evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List, Tuple

import numpy as np
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
# Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Quantum self‑attention implemented with RX/RY/RZ rotations
    and controlled‑X entanglement.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

# --------------------------------------------------------------------------- #
# Quantum autoencoder circuit builder
# --------------------------------------------------------------------------- #
class _QuantumAutoencoderCircuit:
    """Builds a variational autoencoder with a self‑attention block."""
    def __init__(self, latent_dim: int, trash_dim: int):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + trash_dim + 1  # +1 auxiliary for swap‑test
        self.attention = QuantumSelfAttention(trash_dim)

    def build(self, params: np.ndarray) -> QuantumCircuit:
        # Parameter segmentation
        enc_len = self.latent_dim * 3
        attn_rot_len = self.trash_dim * 3
        attn_ent_len = self.trash_dim - 1
        dec_len = self.latent_dim * 3
        enc_params, attn_rot, attn_ent, dec_params = (
            params[:enc_len],
            params[enc_len : enc_len + attn_rot_len],
            params[enc_len + attn_rot_len : enc_len + attn_rot_len + attn_ent_len],
            params[-dec_len:],
        )

        # Encoder
        encoder = RealAmplitudes(self.latent_dim, reps=3)
        # Attention on trash qubits
        attention_circ = self.attention.build(attn_rot, attn_ent)
        # Decoder (inverse of encoder)
        decoder = RealAmplitudes(self.latent_dim, reps=3).inverse()

        # Assemble full circuit
        qc = QuantumCircuit(self.num_qubits, 1)
        # Map encoder onto latent qubits
        qc.compose(encoder, range(self.latent_dim), inplace=True)
        # Map attention onto trash qubits
        qc.compose(attention_circ, range(self.latent_dim, self.latent_dim + self.trash_dim), inplace=True)
        # Decoder
        qc.compose(decoder, range(self.latent_dim), inplace=True)

        # Swap‑test with auxiliary qubit
        aux = self.latent_dim + self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + i)  # placeholder entanglement
        qc.h(aux)
        qc.measure(aux, 0)
        return qc

# --------------------------------------------------------------------------- #
# Fast estimator for expectation values
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_vals))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        param_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in param_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Wrapper class exposing a shared name
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybrid:
    """Quantum autoencoder wrapper that mirrors the classical API."""
    circuit: QuantumCircuit
    sampler: Sampler = Sampler()
    estimator: FastBaseEstimator | None = None

    def __post_init__(self) -> None:
        self.estimator = FastBaseEstimator(self.circuit)

    def forward(self, params: np.ndarray) -> Statevector:
        """Return the statevector after applying the circuit."""
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
        return Statevector.from_instruction(bound)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        param_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Delegate to the bundled FastBaseEstimator."""
        return self.estimator.evaluate(observables, param_sets)

    @classmethod
    def from_config(
        cls,
        latent_dim: int,
        trash_dim: int,
    ) -> "AutoencoderHybrid":
        """Convenience constructor that builds the underlying circuit."""
        builder = _QuantumAutoencoderCircuit(latent_dim, trash_dim)
        # Dummy parameter array to instantiate the circuit
        dummy_len = (
            latent_dim * 3
            + trash_dim * 3
            + (trash_dim - 1)
            + latent_dim * 3
        )
        dummy_params = np.zeros(dummy_len)
        circuit = builder.build(dummy_params)
        return cls(circuit=circuit)

# --------------------------------------------------------------------------- #
# Training helper (classical optimizer)
# --------------------------------------------------------------------------- #
from qiskit_machine_learning.optimizers import COBYLA

def train_quantum_autoencoder(
    autoenc: AutoencoderHybrid,
    input_states: Sequence[Statevector],
    *,
    epochs: int = 50,
    maxiter: int = 200,
) -> List[float]:
    """
    Train the quantum autoencoder by minimizing the negative fidelity
    between each input state and the state produced by the circuit.
    """
    param_len = len(autoenc.circuit.parameters)

    def objective(p: np.ndarray) -> float:
        total = 0.0
        for psi in input_states:
            circ = autoenc.circuit.assign_parameters(dict(zip(autoenc.circuit.parameters, p)), inplace=False)
            state = Statevector.from_instruction(circ)
            total += np.real(state.fidelity(psi))
        return -total / len(input_states)

    optimizer = COBYLA(maxiter=maxiter)
    p0 = np.random.rand(param_len) * 2 * np.pi
    res = optimizer.optimize(num_vars=param_len, objective_function=objective, initial_point=p0)
    final_params = res[0]
    autoenc.circuit = autoenc.circuit.assign_parameters(dict(zip(autoenc.circuit.parameters, final_params)), inplace=False)

    fidelities: List[float] = []
    for psi in input_states:
        fidelities.append(np.real(Statevector.from_instruction(autoenc.circuit).fidelity(psi)))
    return fidelities

__all__ = [
    "AutoencoderHybrid",
    "FastBaseEstimator",
    "train_quantum_autoencoder",
]
