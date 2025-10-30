from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import torch
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

# Helper dataclass for quantum fraud params
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

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool) -> None:
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

# Quantum auto‑encoder circuit
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = qiskit.circuit.library.RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# Quantum quanvolution filter
class QuanvCircuit:
    """A simple quanvolution filter using a random circuit."""
    def __init__(self, kernel_size: int, shots: int = 200, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(row)}
                       for row in data]
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts()
        counts = 0
        for key, val in result.items():
            counts += sum(int(b) for b in key) * val
        return counts / (self.shots * self.n_qubits)

# Quantum sampler QNN
def quantum_sampler_qnn() -> QSamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = Sampler()
    return QSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

@dataclass
class FraudDetectionHybridConfig:
    fraud_input: FraudLayerParameters
    fraud_layers: Sequence[FraudLayerParameters]
    autoencoder_params: dict
    conv_params: dict

class FraudDetectionHybrid:
    """Quantum‑classical fraud detection pipeline mirroring the classical counterpart."""
    def __init__(self, cfg: FraudDetectionHybridConfig) -> None:
        self.fraud_prog = build_fraud_detection_program(cfg.fraud_input, cfg.fraud_layers)
        self.autoencoder = autoencoder_circuit(3, 2)
        self.conv = QuanvCircuit(kernel_size=2, shots=200, threshold=0.5)
        self.sampler = quantum_sampler_qnn()

    def run(self, inputs: np.ndarray) -> np.ndarray:
        # Photonic feature extraction
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        state = eng.run(self.fraud_prog, args=inputs).state
        # Extract a simple classical feature from the state (placeholder)
        feature = np.random.rand(2)
        # Auto‑encoder output (placeholder: use the same feature)
        ae_out = feature
        # Quanvolution filter on flattened data
        conv_out = self.conv.run(ae_out)
        # Combine features for sampler QNN
        qnn_input = torch.tensor([ae_out[0], conv_out], dtype=torch.float32)
        qnn_output = self.sampler(qnn_input)
        return qnn_output.detach().numpy()
