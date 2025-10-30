from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes


# --------------------------------------------------------------------------- #
#  Quantum sub‑modules (adapted from the seed repo)
# --------------------------------------------------------------------------- #

def quantum_fcl(num_qubits: int) -> QuantumCircuit:
    """Parameterised circuit acting as a quantum fully‑connected layer."""
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector("theta", num_qubits)
    qc.h(range(num_qubits))
    qc.barrier()
    for i, p in enumerate(theta):
        qc.ry(p, i)
    qc.measure_all()
    return qc


def quantum_attention(num_qubits: int) -> QuantumCircuit:
    """Quantum self‑attention block."""
    qc = QuantumCircuit(num_qubits)
    rot = ParameterVector("rot", 3 * num_qubits)
    ent = ParameterVector("ent", num_qubits - 1)
    for i in range(num_qubits):
        qc.rx(rot[3 * i], i)
        qc.ry(rot[3 * i + 1], i)
        qc.rz(rot[3 * i + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(ent[i], i, i + 1)
    qc.measure_all()
    return qc


def quantum_autoencoder(num_latent: int, num_trash: int) -> QuantumCircuit:
    """A simple variational auto‑encoder style circuit."""
    qr = qiskit.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = qiskit.ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, qr[: num_latent + num_trash])
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


def quantum_classifier(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """Quantum classifier ansatz mirroring the classical feed‑forward stack."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
#  Hybrid quantum model
# --------------------------------------------------------------------------- #

class HybridQuantumModel:
    """
    Quantum counterpart to HybridModel.  Each classical block is mirrored
    by a parameterised circuit.  The run() method stitches the blocks together
    in the same logical order:
        1. Fully‑connected (FCL) → expectation
        2. Self‑attention → probability distribution
        3. Auto‑encoder → variational output
        4. Classifier → final measurement counts
    """

    def __init__(self, num_qubits: int = 1, depth: int = 2) -> None:
        self.backend = Aer.get_backend("qasm_simulator")

        # 1. FCL
        self.fcl = quantum_fcl(num_qubits)

        # 2. Self‑attention
        self.attn = quantum_attention(num_qubits)

        # 3. Auto‑encoder
        self.autoencoder = quantum_autoencoder(num_latent=3, num_trash=2)

        # 4. Classifier
        self.classifier, _, _, _ = quantum_classifier(num_qubits, depth)

    def _expectation_from_counts(self, counts: dict[str, int]) -> float:
        """Utility: convert measurement counts to an expectation value."""
        states = np.array([int(k, 2) for k in counts.keys()])
        probs = np.array(list(counts.values())) / sum(counts.values())
        return float(np.sum(states * probs))

    def run(
        self,
        fcl_thetas: np.ndarray,
        attn_rot: np.ndarray,
        attn_ent: np.ndarray,
        ae_shots: int = 1024,
        clf_shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the hybrid quantum pipeline.

        Parameters
        ----------
        fcl_thetas : np.ndarray
            Parameters for the fully‑connected layer (one per qubit).
        attn_rot : np.ndarray
            Rotation parameters for the attention circuit (3 * n_qubits).
        attn_ent : np.ndarray
            Entanglement parameters for the attention circuit (n_qubits-1).
        ae_shots : int
            Number of shots for the auto‑encoder.
        clf_shots : int
            Number of shots for the classifier.

        Returns
        -------
        dict[str, int]
            Measurement counts from the final classifier circuit.
        """
        # 1. FCL
        fcl_bound = self.fcl.bind_parameters(ParameterVector("theta", self.fcl.num_parameters))
        fcl_job = execute(fcl_bound, self.backend, shots=1024)
        fcl_counts = fcl_job.result().get_counts()
        fcl_expect = self._expectation_from_counts(fcl_counts)

        # 2. Self‑attention
        attn_bound = self.attn.bind_parameters(
            ParameterVector("rot", self.attn.num_parameters) | ParameterVector("ent", self.attn.num_parameters - 3 * self.attn.num_qubits)
        )
        attn_job = execute(attn_bound, self.backend, shots=attn_shots)
        attn_counts = attn_job.result().get_counts()
        attn_expect = self._expectation_from_counts(attn_counts)

        # 3. Auto‑encoder (variational)
        ae_job = execute(self.autoencoder, self.backend, shots=ae_shots)
        ae_counts = ae_job.result().get_counts()
        ae_expect = self._expectation_from_counts(ae_counts)

        # 4. Classifier – bind the auto‑encoder expectation to the variational parameters
        clf_binding = {p: ae_expect for p in self.classifier.parameters()}
        clf_bound = self.classifier.bind_parameters(clf_binding)
        clf_job = execute(clf_bound, self.backend, shots=clf_shots)
        return clf_job.result().get_counts()
