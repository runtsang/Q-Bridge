import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit

class HybridQuantumLayer:
    """
    Quantum implementation of the hybrid architecture.
    Combines a quanvolution, a quantum auto‑encoder, a simple transformer ansatz
    and a fully‑connected Ry circuit.
    """
    def __init__(self, shots: int = 1024, backend=None):
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_subcircuits()

    def _build_subcircuits(self):
        # 1. Quantum convolution (QuanvCircuit)
        self.conv_circ = self._quantum_conv_circuit(kernel_size=2, threshold=0.0)

        # 2. Quantum auto‑encoder circuit
        self.auto_circ = self._autoencoder_circuit(num_latent=3, num_trash=2)

        # 3. Quantum transformer (simple entangling ansatz)
        self.transformer_circ = self._transformer_circuit(num_qubits=2)

        # 4. Quantum fully‑connected layer (single‑qubit Ry)
        self.fcl_circ = self._fcl_circuit(n_qubits=1)

    def _quantum_conv_circuit(self, kernel_size: int, threshold: float):
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        qc.threshold = threshold
        return qc

    def _autoencoder_circuit(self, num_latent: int, num_trash: int):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _transformer_circuit(self, num_qubits: int):
        qc = QuantumCircuit(num_qubits)
        ansatz = RealAmplitudes(num_qubits, reps=2)
        qc.compose(ansatz, range(num_qubits), inplace=True)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(num_qubits - 1, 0)
        qc.measure_all()
        return qc

    def _fcl_circuit(self, n_qubits: int):
        qc = QuantumCircuit(n_qubits)
        theta = Parameter("theta")
        for q in range(n_qubits):
            qc.ry(theta, q)
        qc.measure_all()
        return qc

    def _run_circuit(self, circ: QuantumCircuit, param_bind: dict):
        job = execute(circ, self.backend, shots=self.shots, parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts(circ)
        total = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        expectation = total / (self.shots * len(circ.qubits))
        return expectation

    def run(self, data: float) -> np.ndarray:
        """
        Run the full quantum pipeline on a scalar input.
        The same scalar is bound to every parameter in all sub‑circuits.
        Returns an array of four expectation values:
        [conv, autoencoder, transformer, fcl].
        """
        conv_bind = {p: data for p in self.conv_circ.parameters}
        auto_bind = {p: data for p in self.auto_circ.parameters}
        trans_bind = {p: data for p in self.transformer_circ.parameters}
        fcl_bind = {self.fcl_circ.parameters[0]: data}

        conv_exp = self._run_circuit(self.conv_circ, conv_bind)
        auto_exp = self._run_circuit(self.auto_circ, auto_bind)
        trans_exp = self._run_circuit(self.transformer_circ, trans_bind)
        fcl_exp = self._run_circuit(self.fcl_circ, fcl_bind)

        return np.array([conv_exp, auto_exp, trans_exp, fcl_exp])

__all__ = ["HybridQuantumLayer"]
