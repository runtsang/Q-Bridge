import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
#  Photonic fraud‑detection parameters
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


# --------------------------------------------------------------------------- #
#  Quantum Kernel Ansatz (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumKernelAnsatz:
    """
    Parameterised circuit that emulates a simple RBF‑style quantum kernel.
    The circuit is identical for every data point pair; the data is encoded
    by setting rotation angles to 0 or π depending on the sign of the input.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = [qiskit.circuit.Parameter(f'theta{i}') for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.params[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 backend,
                 shots: int = 1024) -> float:
        # Encode both vectors into one parameter list
        bind = {}
        for val, p in zip(np.concatenate([x, y]), self.params):
            bind[p] = np.pi if val > 0 else 0
        job = qiskit.execute(self.circuit,
                             backend,
                             shots=shots,
                             parameter_binds=[bind])
        result = job.result().get_counts(self.circuit)
        # Compute probability of measuring |1> on any qubit
        total = sum(result.values())
        ones = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return ones / (shots * self.n_qubits)


# --------------------------------------------------------------------------- #
#  Quantum Self‑Attention (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    A Qiskit implementation of the self‑attention block from the
    reference SelfAttention.py.  Rotation and entangle parameters are
    supplied externally.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, 'q')
        self.cr = qiskit.ClassicalRegister(n_qubits, 'c')

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# --------------------------------------------------------------------------- #
#  Photonic Fraud Detection Program (Strawberry Fields)
# --------------------------------------------------------------------------- #
class PhotonicFraudCircuit:
    """
    Builds a Strawberry Fields program that mirrors the photonic fraud
    detection seed.  The program can be executed on a photonic simulator
    or a real photonic backend.
    """
    def __init__(self, params: FraudLayerParameters):
        self.params = params

    def build(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            BSgate(self.params.bs_theta, self.params.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(self.params.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(self.params.squeeze_r,
                                             self.params.squeeze_phi)):
                Sgate(r, phi) | q[i]
            BSgate(self.params.bs_theta, self.params.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(self.params.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(self.params.displacement_r,
                                             self.params.displacement_phi)):
                Dgate(r, phi) | q[i]
            for i, k in enumerate(self.params.kerr):
                Kgate(k) | q[i]
        return prog


# --------------------------------------------------------------------------- #
#  Hybrid Quantum Convolution Module
# --------------------------------------------------------------------------- #
class HybridQuantumConv:
    """
    Combines a quanvolution circuit, a quantum kernel ansatz, a
    quantum self‑attention block, and a photonic fraud‑detection program.
    All sub‑components are exposed through a common interface.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 1024,
                 threshold: float = 127):
        if backend is None:
            backend = qiskit.Aer.get_backend('qasm_simulator')
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.quanv_circuit = self._build_quanv_circuit(kernel_size)
        self.kernel_ansatz = QuantumKernelAnsatz(self.n_qubits)
        self.self_attention = QuantumSelfAttention(self.n_qubits)

    def _build_quanv_circuit(self, kernel_size):
        n = kernel_size ** 2
        qc = qiskit.QuantumCircuit(n)
        thetas = [qiskit.circuit.Parameter(f'theta{i}') for i in range(n)]
        for i in range(n):
            qc.rx(thetas[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quanvolution circuit on a single 2‑D patch.
        """
        data = np.reshape(data, (1, self.n_qubits))
        binds = []
        for val in data[0]:
            bind = {p: np.pi if val > self.threshold else 0
                    for p in self.quanv_circuit.parameters}
            binds.append(bind)
        job = qiskit.execute(self.quanv_circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=binds)
        result = job.result().get_counts(self.quanv_circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def kernel_matrix(self,
                      a: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix using the quantum kernel ansatz.
        """
        return np.array([[self.kernel_ansatz.evaluate(x, y,
                                                      self.backend,
                                                      self.shots)
                          for y in b] for x in a])

    def attention_counts(self,
                         rotation_params: np.ndarray,
                         entangle_params: np.ndarray) -> dict:
        """
        Run the quantum self‑attention block and return the measurement counts.
        """
        return self.self_attention.run(self.backend,
                                       rotation_params,
                                       entangle_params,
                                       self.shots)

    def photonic_program(self,
                         params: FraudLayerParameters) -> sf.Program:
        """
        Generate a Strawberry Fields program for photonic fraud detection.
        """
        circuit = PhotonicFraudCircuit(params)
        return circuit.build()


__all__ = ["HybridQuantumConv"]
