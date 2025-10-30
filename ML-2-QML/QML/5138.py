import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.circuit.library import RealAmplitudes, CX
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

__all__ = [
    "QuantumKernel",
    "autoencoder_circuit",
    "quantum_sampler_qnn",
    "QuantumKernelAutoencoderFraudSampler",
]

# ------------------------------------------------------------------
# Quantum kernel
# ------------------------------------------------------------------
class QuantumKernel:
    """Variational RBF‑style kernel implemented with Qiskit."""
    def __init__(self, num_wires: int = 4):
        self.num_wires = num_wires
        self.params = ParameterVector("theta", num_wires)
        self.circuit = QuantumCircuit(num_wires)
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.sampler = QiskitSampler()

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel by encoding x forward and y backward."""
        circ = self.circuit.copy()
        for i, val in enumerate(x):
            circ.ry(val, i)
        for i, val in enumerate(y):
            circ.ry(-val, i)
        result = self.sampler.run(circ, shots=1).result()
        counts = result.get_counts()
        # probability of measuring all zeros
        return counts.get("0" * self.num_wires, 0)

# ------------------------------------------------------------------
# Quantum autoencoder circuit
# ------------------------------------------------------------------
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Auxiliary qubit for swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# ------------------------------------------------------------------
# Quantum SamplerQNN
# ------------------------------------------------------------------
def quantum_sampler_qnn(num_qubits: int = 2) -> QiskitSamplerQNN:
    inputs = ParameterVector("input", num_qubits)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(inputs[i], i)
    qc.cx(0, 1)
    for i in range(2):
        qc.ry(weights[i], i)
    qc.cx(0, 1)
    for i in range(2, 4):
        qc.ry(weights[i], i % num_qubits)

    interpret = lambda x: x  # identity mapping

    sampler = QiskitSampler()
    return QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
        interpret=interpret,
        output_shape=2,
    )

# ------------------------------------------------------------------
# Combined quantum wrapper
# ------------------------------------------------------------------
class QuantumKernelAutoencoderFraudSampler:
    """Quantum‑side wrapper mirroring the classical composite."""
    def __init__(self):
        self.quantum_kernel = QuantumKernel()
        self.autoencoder_circ = autoencoder_circuit(3, 2)
        self.sampler_qnn = quantum_sampler_qnn()

    def evaluate_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.quantum_kernel.kernel(x, y)

    def run_sampler(self, latent: np.ndarray) -> np.ndarray:
        """Run the quantum sampler on a latent vector."""
        # The sampler expects a parameter vector; we bind the latent vector
        param_dict = {p: val for p, val in zip(self.sampler_qnn.input_params, latent)}
        result = self.sampler_qnn.get_weights(param_dict)
        return result
