"""Quantum hybrid fraud detection model mirroring the classical implementation.

The quantum version builds a Strawberry Fields photonic program, a Qiskit
quantum convolutional filter, a SamplerQNN auto‑encoder, and an
EstimatorQNN estimator.  The same :class:`FraudLayerParameters` schema is
used to construct the photonic linear mapping, ensuring a one‑to‑one
comparison between the two back‑ends.
"""

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# ----------------------------------------------------------------------
# Parameter schema (shared with the classical model)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Photonic fraud detection program (Strawberry Fields)
# ----------------------------------------------------------------------
def build_photonic_fraud_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program that mimics the classical fraud
    detection pipeline but uses continuous‑variable gates."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

# ----------------------------------------------------------------------
# Quantum convolutional filter (Qiskit)
# ----------------------------------------------------------------------
class QuantumConvFilter:
    """Quantum analogue of the classical Conv filter."""

    def __init__(self, kernel_size: int = 2, shots: int = 200, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array of shape ``(kernel_size, kernel_size)``."""
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in data_flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        counts = sum(sum(int(bit) for bit in key) * val
                     for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

# ----------------------------------------------------------------------
# Quantum auto‑encoder using SamplerQNN
# ----------------------------------------------------------------------
def build_quantum_autoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Construct a Qiskit SamplerQNN that implements an auto‑encoder style
    circuit with a swap‑test based reconstruction loss."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    sampler = Sampler()
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )

# ----------------------------------------------------------------------
# Quantum estimator (EstimatorQNN)
# ----------------------------------------------------------------------
def build_quantum_estimator() -> EstimatorQNN:
    param_in = Parameter("input")
    param_w = Parameter("weight")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(param_in, 0)
    qc.rx(param_w, 0)

    observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y", 1)])
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[param_in],
        weight_params=[param_w],
        estimator=EstimatorQNN,
    )

# ----------------------------------------------------------------------
# Hybrid quantum fraud detector
# ----------------------------------------------------------------------
class FraudDetectorHybrid:
    """Quantum class that mirrors :class:`FraudDetectorHybrid` in the classical
    module.  It orchestrates a photonic fraud program, a quantum convolution
    filter, a SamplerQNN auto‑encoder, and an EstimatorQNN estimator.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        conv_kernel: int = 2,
        conv_shots: int = 200,
        autoencoder_cfg: Tuple[int, int] | None = None,
    ) -> None:
        self.photonic_prog = build_photonic_fraud_program(fraud_params, [])
        self.conv_filter = QuantumConvFilter(kernel_size=conv_kernel,
                                             shots=conv_shots,
                                             threshold=0.5)
        self.autoencoder = build_quantum_autoencoder(
            num_latent=autoencoder_cfg[0] if autoencoder_cfg else 3,
            num_trash=autoencoder_cfg[1] if autoencoder_cfg else 2,
        )
        self.estimator = build_quantum_estimator()

    def predict(self, data: np.ndarray) -> float:
        """Run the full quantum pipeline on a single data sample."""
        # 1. Photonic layer (continuous‑variable)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        # Assume data is a 2‑element vector for the photonic mode
        result = eng.run(self.photonic_prog, data=data)
        photonic_out = result.samples[0][0]  # mock extraction

        # 2. Quantum convolution
        conv_out = self.conv_filter.run(data)

        # 3. Auto‑encoder
        ae_out = self.autoencoder.forward(np.array([conv_out]))
        latent = ae_out[0]

        # 4. Estimator
        pred = self.estimator.forward(latent)
        return float(pred)

    # Simple wrappers for training QNNs – in practice one would use Qiskit
    # training pipelines (e.g., COBYLA) but the skeleton is provided for
    # completeness.
    def train_autoencoder(self, shots: int = 200, epochs: int = 20) -> None:
        pass  # placeholder – training via COBYLA or gradient‑based optimisers

    def train_estimator(self, shots: int = 200, epochs: int = 20) -> None:
        pass  # placeholder

__all__ = [
    "FraudLayerParameters",
    "build_photonic_fraud_program",
    "QuantumConvFilter",
    "build_quantum_autoencoder",
    "build_quantum_estimator",
    "FraudDetectorHybrid",
]
