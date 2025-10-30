"""Hybrid quantum fraud‑detection with attention.

The quantum implementation uses a Qiskit circuit to perform a
self‑attention‑style computation and a Strawberry Fields program
to model the photonic fraud‑detection layers.  The two sub‑circuit
are connected by feeding the measurement outcomes of the Qiskit
attention block into the displacement operations of the photonic
circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# Parameters for fraud‑detection – same dataclass as the classical seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
# Qiskit self‑attention circuit
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Gate‑based self‑attention circuit implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: qiskit.providers.BaseBackend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        circ = self._build_circuit(rotation_params, entangle_params)
        job = execute(circ, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)
        # Convert counts to a flat probability vector over qubits
        probs = np.zeros(self.n_qubits)
        for bitstring, count in counts.items():
            for i, bit in enumerate(bitstring[::-1]):  # reverse to match qubit order
                probs[i] += count * int(bit)
        probs /= shots
        return probs


# --------------------------------------------------------------------------- #
# Strawberry Fields fraud‑detection circuit
# --------------------------------------------------------------------------- #
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Construct a photonic program mirroring the classical fraud layers."""
    prog = sf.Program(2)
    with prog.context as q:
        # Input layer
        _apply_layer(q, input_params, clip=False)
        # Subsequent layers
        for layer in layers:
            _apply_layer(q, layer, clip=True)
        # Final linear readout
        sf.ops.MeasureP(0) | q[0]
        sf.ops.MeasureP(1) | q[1]
    return prog


def _apply_layer(modes: Sequence, params: FraudLayerParameters, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


# --------------------------------------------------------------------------- #
# Hybrid quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionQuantumAttention:
    """Hybrid Qiskit + Strawberry Fields model for fraud detection.

    The forward pass applies a Qiskit self‑attention circuit, extracts
    a probability vector, and uses it to set displacement parameters
    for the photonic fraud‑detection circuit.  The circuit is then
    simulated on a Strawberry Fields engine.
    """

    def __init__(
        self,
        attention_params: tuple[np.ndarray, np.ndarray],
        fraud_params: Iterable[FraudLayerParameters],
        backend: qiskit.providers.BaseBackend | None = None,
    ) -> None:
        # Store parameters
        self.attention_params = attention_params
        self.fraud_params = list(fraud_params)

        # Qiskit attention object
        self.attention = QuantumSelfAttention(n_qubits=attention_params[0].size // 3)
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Pre‑build the photonic program (parameter‑free)
        self.photonic_prog = build_fraud_detection_program(
            self.fraud_params[0], self.fraud_params[1:]
        )
        self.eng = sf.Engine("gaussian")

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Execute the hybrid circuit and return measurement outcomes."""
        # 1️⃣ Qiskit attention step
        probs = self.attention.run(
            self.backend,
            self.attention_params[0],
            self.attention_params[1],
        )

        # 2️⃣ Use the probabilities as displacement amplitudes
        # Map probabilities to the first mode; second mode remains 0
        disp = (probs[0], 0.0)

        # 3️⃣ Apply displacements before the photonic layers
        prog = sf.Program(2)
        with prog.context as q:
            Dgate(disp[0], 0) | q[0]
            Dgate(disp[1], 0) | q[1]
            # Append the pre‑built fraud layers
            for op in self.photonic_prog.operations:
                op.apply(prog.context)

        # 4️⃣ Run the program
        results = self.eng.run(prog, shots=1024)
        return results.samples  # raw quadrature samples

__all__ = ["FraudLayerParameters", "FraudDetectionQuantumAttention"]
