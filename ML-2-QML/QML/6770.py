import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit.circuit.random import random_circuit
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

# --------------------------------------------------------------------------- #
# Quantum‑classical layer parameters – identical to the original QML seed
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

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# --------------------------------------------------------------------------- #
# Photonic layer construction – allows optional override of displacement
# --------------------------------------------------------------------------- #
def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool = True,
    displacement_r_override: Optional[Tuple[float, float]] = None,
) -> None:
    """Apply a single photonic layer to the given modes."""
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    disp_r = displacement_r_override if displacement_r_override else params.displacement_r
    for i, (r, phi) in enumerate(zip(disp_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    displacement_override: Optional[Tuple[float, float]] = None,
) -> sf.Program:
    """Build a Strawberry Fields program with optional displacement override."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False, displacement_r_override=displacement_override)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# Quantum convolution filter – Qiskit implementation (same as Conv.py)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# High‑level hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Hybrid fraud‑detection that runs a Qiskit quanvolution filter on the input
    and feeds the resulting scalar into a Strawberry Fields photonic circuit.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        kernel_size: int = 2,
        conv_threshold: float = 127.0,
        conv_shots: int = 100,
    ) -> None:
        # Quantum convolution
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(kernel_size, backend, conv_shots, conv_threshold)
        # Photonic circuit
        self.input_params = input_params
        self.layers = layers

    def run(self, input_data: np.ndarray) -> float:
        """
        Parameters
        ----------
        input_data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size)

        Returns
        -------
        float
            Prediction value from the photonic circuit
        """
        # 1. Quantum convolution
        conv_out = self.conv.run(input_data)

        # 2. Build photonic program with displacement overridden by convex output
        displacement_override = (conv_out, conv_out)
        program = build_fraud_detection_program(
            self.input_params,
            self.layers,
            displacement_override=displacement_override,
        )

        # 3. Execute on a Fock backend (simulation)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(program, shots=1)
        # The program ends with a linear layer; we return the raw measurement
        return float(result.samples[0][0])

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QuanvCircuit",
    "FraudDetectionHybrid",
]
