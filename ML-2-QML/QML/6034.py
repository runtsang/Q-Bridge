import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuanvCircuit:
    """
    Variational quanvolution circuit that maps a 2‑D image patch
    to a scalar by measuring the mean probability of |1⟩.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127,
                 shots: int = 100,
                 backend: qiskit.providers.Backend = None) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).
        Returns:
            float – mean probability of measuring |1⟩ across all qubits.
        """
        data = np.asarray(data).reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0
                    for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters]) -> sf.Program:
    """
    Build a Strawberry Fields program that mimics the photonic fraud‑detection
    circuit.  The first layer is un‑clipped; subsequent layers are clipped.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence,
                 params: FraudLayerParameters,
                 clip: bool = False) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r,
                                     params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class ConvFraudHybrid:
    """
    Quantum hybrid model that first applies a quanvolution filter to a
    2‑D image patch and then feeds the scalar output into a photonic
    fraud‑detection circuit.  The final result is a single probability
    value suitable for binary classification.
    """
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 127,
                 conv_shots: int = 100,
                 fraud_params: Sequence[FraudLayerParameters] = None,
                 backend: qiskit.providers.Backend = None) -> None:
        self.quanv = QuanvCircuit(kernel_size=conv_kernel_size,
                                   threshold=conv_threshold,
                                   shots=conv_shots,
                                   backend=backend)
        if fraud_params is None:
            fraud_params = []
        # first element is the input layer, rest are hidden layers
        self.fraud_program = build_fraud_detection_program(fraud_params[0],
                                                           fraud_params[1:])
        self.engine = sf.Engine("gaussian_state")

    def run(self, image_patch: np.ndarray) -> float:
        """
        Execute the full hybrid pipeline.
        Returns:
            float – expectation value of the photon number in mode 0.
        """
        conv_output = self.quanv.run(image_patch)  # scalar
        # The photonic circuit expects two modes; we duplicate the scalar
        result = self.engine.run(self.fraud_program,
                                 args={"mode1": conv_output,
                                       "mode2": conv_output})
        # Extract photon‑number expectation from mode 0
        return result.results.expectation_value

__all__ = ["FraudLayerParameters", "QuanvCircuit",
           "build_fraud_detection_program", "ConvFraudHybrid"]
