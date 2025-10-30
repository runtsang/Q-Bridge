from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
# Quantum sub‑blocks
# --------------------------------------------------------------------------- #

def _build_attention_circuit(
    n_qubits: int,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> QuantumCircuit:
    """Quantum self‑attention circuit mirroring the classical version."""
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    for i in range(n_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    for i in range(n_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit


def _build_fcl_circuit(n_qubits: int, thetas: Iterable[float]) -> QuantumCircuit:
    """Parameterised circuit for a single‑output fully‑connected layer."""
    qc = QuantumCircuit(n_qubits)
    theta = Parameter("theta")
    qc.h(range(n_qubits))
    qc.barrier()
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    return qc


def _build_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """A toy auto‑encoder circuit using a swap‑test style measurement."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


def _build_conv_circuit(kernel_size: int, threshold: float) -> QuantumCircuit:
    """Simple quanvolution‑style filter circuit."""
    n_qubits = kernel_size ** 2
    qc = QuantumCircuit(n_qubits)
    theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += qiskit.circuit.random.random_circuit(n_qubits, 2)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #

class HybridSelfAttentionQML:
    """Quantum‑classical hybrid self‑attention pipeline."""
    def __init__(
        self,
        n_qubits: int = 4,
        n_fcl_qubits: int = 1,
        num_latent: int = 3,
        num_trash: int = 2,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 1024,
    ) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.n_qubits = n_qubits
        self.n_fcl_qubits = n_fcl_qubits
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.kernel_size = kernel_size
        self.threshold = threshold

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        fcl_thetas: Iterable[float],
        inputs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Execute the full hybrid pipeline and return structured outputs."""
        # 1. Self‑attention
        attn_circ = _build_attention_circuit(self.n_qubits, rotation_params, entangle_params)
        job = qiskit.execute(attn_circ, self.backend, shots=self.shots)
        attn_counts = job.result().get_counts(attn_circ)

        # 2. Fully‑connected layer
        fcl_circ = _build_fcl_circuit(self.n_fcl_qubits, fcl_thetas)
        theta = Parameter("theta")
        bind = {theta: list(fcl_thetas)[0]}
        job = qiskit.execute(fcl_circ.bind_parameters(bind), self.backend, shots=self.shots)
        fcl_counts = job.result().get_counts(fcl_circ)

        # 3. Auto‑encoder
        ae_circ = _build_autoencoder_circuit(self.num_latent, self.num_trash)
        job = qiskit.execute(ae_circ, self.backend, shots=self.shots)
        ae_counts = job.result().get_counts(ae_circ)

        # 4. Convolution filter
        conv_circ = _build_conv_circuit(self.kernel_size, self.threshold)
        data = np.reshape(inputs, (1, self.kernel_size ** 2))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[conv_circ.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(conv_circ, self.backend, shots=self.shots, parameter_binds=param_binds)
        conv_counts = job.result().get_counts(conv_circ)

        # Aggregate results
        return {
            "attention_counts": attn_counts,
            "fcl_counts": fcl_counts,
            "autoencoder_counts": ae_counts,
            "conv_counts": conv_counts,
        }


def SelfAttention() -> HybridSelfAttentionQML:
    """Factory matching the original interface."""
    return HybridSelfAttentionQML()

__all__ = ["SelfAttention", "HybridSelfAttentionQML"]
