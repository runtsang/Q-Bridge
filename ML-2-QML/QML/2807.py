"""Quantum hybrid attention + fraud‑detection pipeline.

The module defines a `QuantumSelfAttentionFraud` class that
constructs a composite circuit:
  1. A Qiskit self‑attention sub‑circuit (rotations + controlled‑Rx).
  2. Measurement of the qubits to obtain a probability distribution.
  3. A mapping from measurement counts to parameters for a
     Strawberry Fields photonic fraud‑detection program.
  4. Execution of the photonic program on a simulator.
The design follows the structure of the original seeds while
introducing a bridge between qubit and photonic domains.

Typical usage:

```python
from SelfAttention__gen064 import QuantumSelfAttentionFraud
backend = qiskit.Aer.get_backend("qasm_simulator")
qmodel = QuantumSelfAttentionFraud(n_qubits=4, backend=backend)
counts = qmodel.run_attention(rotation_params, entangle_params)
results = qmodel.run_photonic(counts)
```
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1. Quantum self‑attention sub‑circuit (adapted from the original seed)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Basic Qiskit self‑attention block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024):
        qc = self._build(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

# --------------------------------------------------------------------------- #
# 2. Photonic fraud‑detection program (adapted from the original seed)
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

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: list[FraudLayerParameters]) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

def _apply_layer(modes, params: FraudLayerParameters, clip: bool):
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# --------------------------------------------------------------------------- #
# 3. Hybrid class that stitches the two domains together
# --------------------------------------------------------------------------- #
class QuantumSelfAttentionFraud:
    """Composite quantum model that runs a self‑attention sub‑circuit
    and then uses the measurement outcomes to parameterise a
    photonic fraud‑detection program.
    """
    def __init__(self, n_qubits: int = 4, backend=None):
        self.attention = QuantumSelfAttention(n_qubits)
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        # Example photonic parameters – in practice these would be learned.
        self.input_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.5, 0.5),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        self.layer_params = [self.input_params for _ in range(3)]

    def _map_counts_to_params(self, counts: dict) -> list[FraudLayerParameters]:
        """
        Simple heuristic: convert measurement outcome frequencies into
        small perturbations of the photonic parameters.
        """
        total = sum(counts.values())
        probs = {int(k, 2): v / total for k, v in counts.items()}
        # Map the first few bits to a tiny shift in bs_theta
        shifts = []
        for i in range(len(self.layer_params)):
            shift = probs.get(i, 0.0) * 0.05  # small perturbation
            params = FraudLayerParameters(
                bs_theta=self.input_params.bs_theta + shift,
                bs_phi=self.input_params.bs_phi,
                phases=self.input_params.phases,
                squeeze_r=self.input_params.squeeze_r,
                squeeze_phi=self.input_params.squeeze_phi,
                displacement_r=self.input_params.displacement_r,
                displacement_phi=self.input_params.displacement_phi,
                kerr=self.input_params.kerr,
            )
            shifts.append(params)
        return shifts

    def run_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
                      shots: int = 1024) -> dict:
        return self.attention.run(self.backend, rotation_params, entangle_params, shots)

    def run_photonic(self, counts: dict) -> np.ndarray:
        """Execute the photonic program parameterised by the measurement
        counts and return the output state vector."""
        photonic_params = self._map_counts_to_params(counts)
        prog = build_fraud_detection_program(self.input_params, photonic_params)
        eng = sf.Engine("gaussian", backend="gaussian_state")
        result = eng.run(prog)
        return result.state

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """Convenience wrapper that runs the full pipeline."""
        counts = self.run_attention(rotation_params, entangle_params, shots)
        return self.run_photonic(counts)

__all__ = ["QuantumSelfAttentionFraud", "QuantumSelfAttention", "build_fraud_detection_program", "FraudLayerParameters"]
