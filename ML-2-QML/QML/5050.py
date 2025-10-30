from __future__ import annotations

import numpy as np
from typing import Tuple, Sequence, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit import execute

# --------------------------------------------------------------------------- #
#  Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFullyConnectedLayer:
    """One‑qubit parameterised circuit that mimics the classical FCL."""
    def __init__(self, n_qubits: int = 1):
        self.backend = AerSimulator()
        self.shots = 1000
        self.qc = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", n_qubits)
        self.qc.h(range(n_qubits))
        for i in range(n_qubits):
            self.qc.ry(self.theta[i], i)
        self.qc.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        param_binds = [{self.theta[i]: t for i, t in enumerate(thetas)}]
        job = execute(self.qc, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.qc)
        probs = np.array(list(counts.values())) / self.shots
        expectation = probs.sum()
        return np.array([expectation])

def FCL() -> QuantumFullyConnectedLayer:
    return QuantumFullyConnectedLayer()

# --------------------------------------------------------------------------- #
#  Quantum auto‑encoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderQCL:
    """Variational auto‑encoder with swap‑test latent reconstruction and optional quantum layer."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 use_quantum_layer: bool = False):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.use_quantum_layer = use_quantum_layer
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        if use_quantum_layer:
            self.qcl = FCL()

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, data: np.ndarray) -> np.ndarray:
        job = self.sampler.run(self.circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        prob = sum(counts.values()) / 1024
        return np.array([prob])

    def forward(self, data: np.ndarray) -> np.ndarray:
        latent = self.encode(data)
        if self.use_quantum_layer:
            latent = self.qcl.run(latent)
        return latent

# --------------------------------------------------------------------------- #
#  Quantum classifier circuit builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Sequence[ParameterVector],
                                                  Sequence[ParameterVector],
                                                  List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
#  Quantum Quanvolution (placeholder)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter:
    """Quantum filter that applies a random kernel to 2×2 patches."""
    def __init__(self):
        self.n_wires = 4

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Placeholder: return zeros with expected shape
        batch = x.shape[0]
        return np.zeros((batch, 4 * 14 * 14))

class QuanvolutionClassifier:
    def __init__(self):
        self.qfilter = QuanvolutionFilter()
        self.linear = np.eye(10)

    def forward(self, x: np.ndarray) -> np.ndarray:
        feat = self.qfilter.forward(x)
        logits = self.linear @ feat.T
        return logits

__all__ = ["HybridAutoencoderQCL",
           "build_classifier_circuit",
           "QuanvolutionFilter",
           "QuanvolutionClassifier",
           "FCL"]
