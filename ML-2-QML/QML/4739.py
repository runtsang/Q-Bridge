import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import RealAmplitudes

class HybridFCL:
    """
    Quantum counterpart of the classical HybridFCL.  The circuit first
    runs a self‑attention style block, then a quantum auto‑encoder
    with a domain‑wall and swap‑test, and finally returns the
    expectation value of the auxiliary qubit.
    """

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build component circuits
        self.attention_circuit = self._build_attention()
        self.autoencoder_circuit = self._build_autoencoder()

        # Compose into a single executable circuit
        self.combined_circuit = self._compose()

        # Parameter ordering
        self.attn_params = list(self.attention_circuit.parameters)
        self.auto_params = list(self.autoencoder_circuit.parameters)

    def _build_attention(self):
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)

        # Rotations
        for i in range(self.n_qubits):
            circ.rx(qiskit.circuit.Parameter(f"rot_{3 * i}"), i)
            circ.ry(qiskit.circuit.Parameter(f"rot_{3 * i + 1}"), i)
            circ.rz(qiskit.circuit.Parameter(f"rot_{3 * i + 2}"), i)

        # Entangling CRX
        for i in range(self.n_qubits - 1):
            circ.crx(qiskit.circuit.Parameter(f"ent_{i}"), i, i + 1)

        circ.measure(qr, cr)
        return circ

    def _build_autoencoder(self):
        num_latent, num_trash = 3, 2
        total = num_latent + 2 * num_trash + 1
        qr = QuantumRegister(total, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Ansatz (RealAmplitudes)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        circ.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        circ.barrier()

        # Domain‑wall on trash qubits
        for i in range(num_trash):
            circ.x(num_latent + 2 * num_trash - i)

        # Swap‑test with auxiliary qubit
        aux = num_latent + 2 * num_trash
        circ.h(aux)
        for i in range(num_trash):
            circ.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])
        return circ

    def _compose(self):
        # Total number of qubits = attention + autoencoder
        total_qubits = self.n_qubits + self.autoencoder_circuit.num_qubits
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        combined = QuantumCircuit(qr, cr)

        # Place attention on first block
        combined.compose(self.attention_circuit, range(self.n_qubits), inplace=True)

        # Place auto‑encoder after it
        combined.compose(
            self.autoencoder_circuit,
            range(self.n_qubits, self.n_qubits + self.autoencoder_circuit.num_qubits),
            inplace=True,
        )
        return combined

    def run(self, thetas: np.ndarray):
        """
        Execute the hybrid circuit.

        Parameters
        ----------
        thetas : array‑like
            Concatenated parameters for the attention block
            (3*n_qubits + n_qubits‑1 elements) followed by the
            auto‑encoder ansatz angles.

        Returns
        -------
        np.ndarray
            Expectation value of the measured auxiliary qubit.
        """
        # Split parameters
        n_att = len(self.attn_params)
        att_thetas = thetas[:n_att]
        auto_thetas = thetas[n_att:]

        # Build parameter binding
        bind = {self.attn_params[i]: att_thetas[i] for i in range(n_att)}
        bind.update({self.auto_params[i]: auto_thetas[i] for i in range(len(auto_thetas))})

        job = execute(
            self.combined_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self.combined_circuit)

        # Expectation of the auxiliary qubit (bit 0)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridFCL"]
